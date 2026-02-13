import torch
import time
import numpy as np

# main.py에서 필요한 컴포넌트들을 가져옵니다.
# (원래는 별도의 공유 라이브러리로 분리하는 것이 가장 이상적입니다.)
from main import (
    PINN,
    TrainingConfig,
    physics_loss_helmholtz,
    resample_collocation_points,
    update_adaptive_weights
)

# --- 벤치마크용 학습 함수 ---
def run_training_case(config: TrainingConfig, target_loss: float):
    """주어진 설정으로 학습을 실행하고 수렴 시간과 최종 손실을 반환합니다."""
    
    start_time = time.monotonic()
    
    # 디바이스 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # 모델 및 옵티마이저 초기화
    pinn_model = PINN().to(device)
    optimizer = torch.optim.Adam(pinn_model.parameters(), lr=config.lr)
    loss_fn = nn.MSELoss()

    # 가중치 초기화
    adaptive_weights = {'bc': torch.tensor(1.0, device=device), 'pde': torch.tensor(1.0, device=device)}

    # 경계 조건 및 초기 콜로케이션 포인트 설정
    x_min, x_max = config.domain_size[0]
    y_min, y_max = config.domain_size[1]
    edge_x = torch.linspace(x_min, x_max, 256, device=device).view(-1, 1)
    edge_y = torch.linspace(y_min, y_max, 256, device=device).view(-1, 1)
    bc_points = torch.cat([
        torch.cat([edge_x, torch.full_like(edge_x, y_min)], dim=1),
        torch.cat([edge_x, torch.full_like(edge_x, y_max)], dim=1),
        torch.cat([torch.full_like(edge_y, x_min), edge_y], dim=1),
        torch.cat([torch.full_like(edge_y, x_max), edge_y], dim=1)
    ], dim=0)
    bc_values = torch.zeros(bc_points.shape[0], 1, device=device)
    collocation_points = (torch.rand(config.num_collocation_points, 2, device=device) * 2 - 1) * torch.tensor([x_max, y_max], device=device)

    # 학습 루프
    final_epoch = config.epochs
    for epoch in range(config.epochs + 1):
        pinn_model.train()
        
        # 중요도 샘플링
        if config.use_importance_sampling and epoch > 0 and epoch % 1000 == 0:
            collocation_points = resample_collocation_points(pinn_model, config)

        # 손실 계산
        loss_bc = loss_fn(pinn_model(bc_points), bc_values)
        loss_pde = physics_loss_helmholtz(pinn_model, collocation_points, config.k)

        # 동적 가중치 업데이트
        if config.use_adaptive_weighting and epoch > 0 and epoch % 500 == 0:
            adaptive_weights = update_adaptive_weights(pinn_model, loss_bc, loss_pde, optimizer, adaptive_weights)

        # 가중 손실 적용
        total_loss = (adaptive_weights['bc'] * loss_bc) + (adaptive_weights['pde'] * loss_pde)
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # 수렴 조건 확인
        if total_loss.item() < target_loss:
            final_epoch = epoch
            break
        
        # 로그 (간소화)
        if epoch % 2000 == 0:
            print(f"  Epoch {epoch}/{config.epochs}, Loss: {total_loss.item():.6f}")

    end_time = time.monotonic()
    
    return {
        "time_taken": end_time - start_time,
        "final_loss": total_loss.item(),
        "converged_at_epoch": final_epoch
    }

# --- 메인 벤치마크 실행 ---
if __name__ == "__main__":
    TARGET_LOSS = 0.001
    BASE_CONFIG = {
        "k": 2.5,
        "epochs": 20000,
        "lr": 0.001,
        "num_collocation_points": 4096,
        "domain_size": [[-1, 1], [-1, 1]],
        "resolution": 64,
        "model_id": "benchmark_run"
    }

    benchmark_cases = {
        "1. Baseline": TrainingConfig(
            **BASE_CONFIG,
            use_importance_sampling=False,
            use_adaptive_weighting=False
        ),
        "2. + Importance Sampling": TrainingConfig(
            **BASE_CONFIG,
            use_importance_sampling=True,
            use_adaptive_weighting=False
        ),
        "3. + All Features": TrainingConfig(
            **BASE_CONFIG,
            use_importance_sampling=True,
            use_adaptive_weighting=True
        ),
    }

    results = {}

    print("=" * 60)
    print(f"PINN Training Benchmark (Target Loss: {TARGET_LOSS})")
    print("=" * 60)

    for name, config in benchmark_cases.items():
        print(f"
Running case: {name}...")
        results[name] = run_training_case(config, TARGET_LOSS)
        print(f"Completed in {results[name]['time_taken']:.2f} seconds.")

    # --- 최종 리포트 출력 ---
    print("

" + "=" * 60)
    print("Benchmark Results")
    print("-" * 60)
    print(f"{'Case':<25} | {'Time (s)':<12} | {'Final Loss':<15} | {'Epochs'}")
    print("-" * 60)

    baseline_time = results["1. Baseline"]["time_taken"]

    for name, result in results.items():
        time_str = f"{result['time_taken']:.2f}"
        loss_str = f"{result['final_loss']:.6f}"
        epoch_str = f"{result['converged_at_epoch']}"

        if name != "1. Baseline":
            improvement = ((baseline_time - result['time_taken']) / baseline_time) * 100
            time_str += f" ({'+' if improvement < 0 else ''}{improvement:.1f}%)"

        print(f"{name:<25} | {time_str:<12} | {loss_str:<15} | {epoch_str}")
    
    print("-" * 60)
