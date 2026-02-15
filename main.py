import asyncio
import torch
import torch.nn as nn
import numpy as np
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import uuid
import time

# 로컬 모듈 import
from numerical_solver import solve_wave_equation_2d_fdtd
from supabase_client import get_supabase_client, is_connected
# analysis.py에서 분석 함수들 import
from analysis import (
    fetch_latest_run_logs,
    analyze_learning_dynamics,
    calculate_complexity,
    generate_abstracts
)

# --- FastAPI 앱 초기화 ---
app = FastAPI()

# --- GPU 가속 설정 ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- PINN 모델 정의 (3D 입력) ---
class PINN(nn.Module):
    def __init__(self, layers=[3, 64, 64, 64, 64, 1]):
        super(PINN, self).__init__()
        self.net = nn.Sequential()
        for i in range(len(layers) - 1):
            self.net.add_module(f"linear_{i}", nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2:
                self.net.add_module(f"tanh_{i}", nn.Tanh())
    def forward(self, xyt):
        return self.net(xyt)

# --- (The rest of the PINN training logic remains the same) ---
# ... (get_pde_residual_wave, physics_loss_wave, etc. are unchanged) ...
def get_pde_residual_wave(model, points, c):
    points.requires_grad_(True)
    u = model(points)
    grad_u = torch.autograd.grad(u, points, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x, u_y, u_t = grad_u[:, 0:1], grad_u[:, 1:2], grad_u[:, 2:3]
    u_xx = torch.autograd.grad(u_x, points, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, points, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    u_tt = torch.autograd.grad(u_t, points, grad_outputs=torch.ones_like(u_t), create_graph=True)[0][:, 2:3]
    return u_tt - (c**2 * (u_xx + u_yy))
def physics_loss_wave(model, points, c):
    residual = get_pde_residual_wave(model, points, c)
    return torch.mean(residual**2)
def initial_condition_loss(model, ic_points, initial_u_values, initial_u_t_values):
    u_pred = model(ic_points)
    loss_u = torch.mean((u_pred - initial_u_values)**2)
    u_t_pred = torch.autograd.grad(u_pred, ic_points, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0][:, 2:3]
    loss_u_t = torch.mean((u_t_pred - initial_u_t_values)**2)
    return loss_u + loss_u_t
def resample_collocation_points_3d(model, config):
    num_candidates = config.num_collocation_points * 10
    x_rand = torch.rand(num_candidates, 1, device=device) * (config.domain_size[0][1] - config.domain_size[0][0]) + config.domain_size[0][0]
    y_rand = torch.rand(num_candidates, 1, device=device) * (config.domain_size[1][1] - config.domain_size[1][0]) + config.domain_size[1][0]
    t_rand = torch.rand(num_candidates, 1, device=device) * (config.time_domain[1] - config.time_domain[0]) + config.time_domain[0]
    candidate_points = torch.cat([x_rand, y_rand, t_rand], dim=1)
    model.eval()
    with torch.no_grad():
        residuals = get_pde_residual_wave(model, candidate_points, config.c)
    model.train()
    weights = torch.abs(residuals.squeeze()) + 1e-8
    probabilities = weights / torch.sum(weights)
    sampled_indices = torch.multinomial(probabilities, config.num_collocation_points, replacement=True)
    return candidate_points[sampled_indices]
def update_adaptive_weights_3loss(model, losses, optimizer, last_weights, momentum=0.9):
    grad_norms = {}
    for name, loss in losses.items():
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        grad_norms[name] = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm()
    optimizer.zero_grad()
    mean_grad_norm = torch.mean(torch.stack(list(grad_norms.values())))
    new_weights = {}
    for name, norm in grad_norms.items():
        weight = mean_grad_norm / (norm + 1e-8)
        new_weights[name] = momentum * last_weights[name] + (1 - momentum) * weight
    return new_weights
def generate_pinn_wavefield_at_t(model, domain_size, resolution, t):
    x = torch.linspace(domain_size[0][0], domain_size[0][1], resolution, device=device)
    y = torch.linspace(domain_size[1][0], domain_size[1][1], resolution, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    t_grid = torch.full_like(xx, t)
    xyt = torch.stack([xx.flatten(), yy.flatten(), t_grid.flatten()], dim=-1)
    model.eval()
    with torch.no_grad():
        u_pred = model(xyt).reshape(resolution, resolution)
    model.train()
    return u_pred.cpu().numpy()
class TrainingConfig(BaseModel):
    c: float = 1.0
    epochs: int = 20000
    lr: float = 0.001
    num_collocation_points: int = 8192
    num_boundary_points: int = 2048
    num_initial_points: int = 4096
    domain_size: list = [[-1, 1], [-1, 1]]
    time_domain: list = [0, 2]
    resolution: int = 51
    use_importance_sampling: bool = True
    use_adaptive_weighting: bool = True
async def log_to_supabase(data):
    if is_connected():
        try:
            get_supabase_client().table("training_logs").insert(data).execute()
        except Exception as e:
            print(f"Epoch {data.get('epoch', 'N/A')}: Failed to log to Supabase. Error: {e}")
async def train_pinn_wave(websocket: WebSocket, config: TrainingConfig):
    run_id = str(uuid.uuid4())
    pinn_model = PINN().to(device)
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()
    adaptive_weights = {'bc': torch.tensor(1.0), 'ic': torch.tensor(1.0), 'pde': torch.tensor(1.0)}
    await websocket.send_text(json.dumps({"status": "solving_fdtd", "run_id": run_id}))
    fdtd_steps = int(config.time_domain[1] / ((1/config.c) * (1/np.sqrt(1/((config.domain_size[0][1]-config.domain_size[0][0])/config.resolution)**2 * 2)) * 0.5))
    u_fdtd = solve_wave_equation_2d_fdtd(grid_points=config.resolution, c=config.c, time_steps=fdtd_steps)
    await websocket.send_text(json.dumps({"status": "fdtd_solved"}))
    ic_points_spatial = (torch.rand(config.num_initial_points, 2, device=device) * 2 - 1)
    ic_points = torch.cat([ic_points_spatial, torch.zeros(config.num_initial_points, 1, device=device)], dim=1)
    initial_u = torch.exp(-torch.sum(ic_points_spatial**2 / (2 * 0.1**2), dim=1, keepdim=True))
    initial_u_t = torch.zeros_like(initial_u)
    await websocket.send_text(json.dumps({"status": "starting_pinn_training", "config": config.dict()}))
    for epoch in range(config.epochs + 1):
        pinn_model.train()
        t_bc = torch.rand(config.num_boundary_points, 1, device=device) * config.time_domain[1]
        x_bc_wall = torch.rand(config.num_boundary_points // 2, 1, device=device) * 2 - 1
        y_bc_wall = torch.rand(config.num_boundary_points // 2, 1, device=device) * 2 - 1
        x_bc = torch.cat([x_bc_wall, torch.ones_like(x_bc_wall), x_bc_wall, -torch.ones_like(x_bc_wall)], dim=0)
        y_bc = torch.cat([torch.ones_like(y_bc_wall), y_bc_wall, -torch.ones_like(y_bc_wall), y_bc_wall], dim=0)
        bc_points = torch.cat([x_bc, y_bc, torch.cat([t_bc]*4, dim=0)], dim=1)
        bc_values = torch.zeros(bc_points.shape[0], 1, device=device)
        if config.use_importance_sampling and epoch > 0 and epoch % 1000 == 0:
            await websocket.send_text(json.dumps({"status": f"resampling_points_at_epoch_{epoch}"}))
            collocation_points = resample_collocation_points_3d(pinn_model, config)
        else:
             collocation_points = torch.cat([torch.rand(config.num_collocation_points, 2, device=device) * 2 - 1, torch.rand(config.num_collocation_points, 1, device=device) * config.time_domain[1]], dim=1)
        loss_bc = loss_fn(pinn_model(bc_points), bc_values)
        loss_ic = initial_condition_loss(pinn_model, ic_points, initial_u, initial_u_t)
        loss_pde = physics_loss_wave(pinn_model, collocation_points, config.c)
        if config.use_adaptive_weighting and epoch > 0 and epoch % 500 == 0:
            adaptive_weights = update_adaptive_weights_3loss(pinn_model, {'bc': loss_bc, 'ic': loss_ic, 'pde': loss_pde}, optimizer, adaptive_weights)
        total_loss = (adaptive_weights['bc'] * loss_bc) + (adaptive_weights['ic'] * loss_ic) + (adaptive_weights['pde'] * loss_pde)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if torch.isnan(total_loss) or total_loss.item() > 5e5: return
        if epoch % 250 == 0:
            t_slice = torch.rand(1).item() * config.time_domain[1]
            pinn_wavefield = generate_pinn_wavefield_at_t(pinn_model, config.domain_size, config.resolution, t_slice)
            fdtd_frame_index = min(int(t_slice / (config.time_domain[1] / fdtd_steps)), fdtd_steps - 1)
            fdtd_wavefield = u_fdtd[fdtd_frame_index]
            log_data = {"epoch": epoch, "total_loss": total_loss.item(), "loss_bc": loss_bc.item(), "loss_ic": loss_ic.item(), "loss_pde": loss_pde.item()}
            await log_to_supabase({"model_id": run_id, **log_data, "status": "in_progress"})
            log_data.update({"wavefield_prediction": pinn_wavefield.tolist(), "wavefield_ground_truth": fdtd_wavefield.tolist(), "time_slice": t_slice, "weights": {k: v.item() for k,v in adaptive_weights.items()}})
            try:
                await websocket.send_text(json.dumps(log_data))
            except WebSocketDisconnect: return
    await websocket.send_text(json.dumps({"status": "training_completed"}))
    print("Training finished.")

# --- WebSocket 엔드포인트 ---
@app.websocket("/ws/pinn-train")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        message = await websocket.receive_text()
        config_data = json.loads(message)
        config = TrainingConfig(**config_data)
        await train_pinn_wave(websocket, config)
    except WebSocketDisconnect:
        print("Client disconnected from /ws/pinn-train")
    except Exception as e:
        print(f"An error occurred in WebSocket: {e}")
        await websocket.send_text(json.dumps({"error": "An internal server error occurred."}))
    finally:
        if websocket.client_state.name == 'CONNECTED':
             await websocket.close()

# --- 리포트 생성 API 엔드포인트 ---
@app.get("/api/report/latest")
async def get_latest_report():
    """Generates and returns a full analysis report for the latest training run."""
    df_logs, config = fetch_latest_run_logs()
    
    if df_logs is None:
        raise HTTPException(status_code=404, detail="No training logs found to generate a report.")

    dynamics_results = analyze_learning_dynamics(df_logs)
    complexity_results = calculate_complexity(pde_type='2D Wave Equation')

    report_data = {
        "model_id": df_logs['model_id'].iloc[0],
        "config": config,
        "learning_dynamics": dynamics_results,
        "complexity": complexity_results,
    }
    
    abstracts = generate_abstracts(report_data)
    report_data['abstracts'] = abstracts
    
    # To-Do: Save to `research_reports` table when schema is known
    
    return report_data


@app.get("/")
def read_root():
    return {"message": "PINN WaveLab FastAPI Server is running. Connect to /ws/pinn-train to start training."}
