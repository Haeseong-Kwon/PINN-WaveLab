import asyncio
import torch
import torch.nn as nn
import numpy as np
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uuid

# 로컬 모듈 import
from numerical_solver import solve_helmholtz_2d_fdm
from supabase_client import get_supabase_client, is_connected

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- GPU 가속 설정 ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# --- PINN 모델 정의 (2D 입력) ---
class PINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 1]):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f"tanh_{i}", nn.Tanh())
    def forward(self, xy):
        return self.net(xy)

# --- Physics Loss (2D Helmholtz) ---
def physics_loss_helmholtz(model, points, k):
    u = model(points)
    grad_u = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = grad_u[:, 0:1], grad_u[:, 1:2]
    u_xx = torch.autograd.grad(u_x, points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, points, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    residual = u_xx + u_yy + k**2 * u
    return nn.MSELoss()(residual, torch.zeros_like(residual))

# PINN 예측 결과 생성
def generate_pinn_wavefield(model, domain_size, resolution):
    x_min, x_max = domain_size[0]
    y_min, y_max = domain_size[1]
    x = torch.linspace(x_min, x_max, resolution, device=device)
    y = torch.linspace(y_min, y_max, resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    xy = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    model.eval()
    with torch.no_grad():
        u_pred = model(xy).reshape(resolution, resolution)
    model.train()
    return u_pred.cpu().numpy()

# --- 데이터 모델 ---
class TrainingConfig(BaseModel):
    k: float = 1.0
    epochs: int = 10000
    lr: float = 0.001
    num_collocation_points: int = 4096
    domain_size: list = [[-1, 1], [-1, 1]]
    resolution: int = 64
    model_id: str = None # Supabase 'pinn_models' 테이블의 ID와 연결

# --- Supabase 로깅 함수 ---
async def log_to_supabase(data):
    if is_connected():
        try:
            get_supabase_client().table("training_logs").insert(data).execute()
        except Exception as e:
            print(f"Epoch {data.get('epoch', 'N/A')}: Failed to log to Supabase. Error: {e}")

# --- 메인 학습 함수 (비동기, 실시간 스트리밍) ---
async def train_pinn_2d(websocket: WebSocket, config: TrainingConfig):
    run_id = str(uuid.uuid4()) # 개별 학습 세션을 위한 고유 ID
    pinn_model = PINN().to(device)
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    # Ground Truth (FDM) 계산
    await websocket.send_text(json.dumps({"status": "solving_fdm", "run_id": run_id}))
    _, _, u_fdm = solve_helmholtz_2d_fdm(k=config.k, domain_size=tuple(map(tuple, config.domain_size)), grid_points=config.resolution, boundary_conditions={'type': 'dirichlet', 'value': 0})
    await websocket.send_text(json.dumps({"status": "fdm_solved", "wavefield_ground_truth": u_fdm.tolist()}))

    # 경계 조건 및 Collocation Points 설정
    x_min, x_max = config.domain_size[0]
    y_min, y_max = config.domain_size[1]
    edge_x = torch.linspace(x_min, x_max, 256, device=device).view(-1, 1)
    edge_y = torch.linspace(y_min, y_max, 256, device=device).view(-1, 1)
    bc_points = torch.cat([torch.cat([edge_x, torch.full_like(edge_x, y_min)], dim=1), torch.cat([edge_x, torch.full_like(edge_x, y_max)], dim=1), torch.cat([torch.full_like(edge_y, x_min), edge_y], dim=1), torch.cat([torch.full_like(edge_y, x_max), edge_y], dim=1)], dim=0)
    bc_values = torch.zeros(bc_points.shape[0], 1, device=device)
    collocation_points = (torch.rand(config.num_collocation_points, 2, device=device) * 2 - 1) * torch.tensor([x_max, y_max], device=device)
    collocation_points.requires_grad = True

    await websocket.send_text(json.dumps({"status": "starting_pinn_training", "config": config.dict()}))

    for epoch in range(config.epochs + 1):
        optimizer.zero_grad()
        loss_bc = loss_fn(pinn_model(bc_points), bc_values)
        loss_pde = physics_loss_helmholtz(pinn_model, collocation_points, config.k)
        total_loss = loss_bc + loss_pde
        total_loss.backward()
        optimizer.step()

        # Divergence Check
        if torch.isnan(total_loss) or total_loss.item() > 5e5:
            print(f"Divergence detected at epoch {epoch}. Loss: {total_loss.item()}")
            await websocket.send_text(json.dumps({"status": "diverged", "epoch": epoch, "loss": total_loss.item()}))
            await log_to_supabase({"model_id": config.model_id, "epoch": epoch, "total_loss": total_loss.item(), "status": "diverged", "training_config": config.dict()})
            return

        if epoch % 200 == 0:
            pinn_wavefield = generate_pinn_wavefield(pinn_model, config.domain_size, config.resolution)
            log_data = {"epoch": epoch, "total_loss": total_loss.item(), "boundary_loss": loss_bc.item(), "physics_loss": loss_pde.item()}
            
            await log_to_supabase({"model_id": config.model_id, **log_data, "status": "in_progress", "training_config": config.dict()})
            
            log_data["wavefield_prediction"] = pinn_wavefield.tolist() # Add wavefield for websocket
            try:
                await websocket.send_text(json.dumps(log_data))
            except WebSocketDisconnect:
                print(f"Client disconnected. Stopping training at epoch {epoch}.")
                torch.save(pinn_model.state_dict(), "pinn_helmholtz_2d_aborted.pth")
                await log_to_supabase({"model_id": config.model_id, "epoch": epoch, "total_loss": total_loss.item(), "status": "aborted", "training_config": config.dict()})
                return

    torch.save(pinn_model.state_dict(), "pinn_helmholtz_2d_final.pth")
    await websocket.send_text(json.dumps({"status": "training_completed"}))
    await log_to_supabase({"model_id": config.model_id, "epoch": config.epochs, "total_loss": total_loss.item(), "status": "completed", "training_config": config.dict()})
    print("Training finished.")

# --- WebSocket 엔드포인트 ---
@app.websocket("/ws/pinn-train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        message = await websocket.receive_text()
        config_data = json.loads(message)
        config = TrainingConfig(**config_data)
        await train_pinn_2d(websocket, config)
    except WebSocketDisconnect:
        print("Client disconnected from /ws/pinn-train")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
        await websocket.send_text(json.dumps({"error": "An internal server error occurred."}))
    finally:
        if websocket.client_state.name == 'CONNECTED':
             await websocket.close()

@app.get("/")
def read_root():
    return {"message": "PINN WaveLab FastAPI Server is running. Connect to /ws/pinn-train to start training."}
