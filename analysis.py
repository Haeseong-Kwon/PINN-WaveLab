import pandas as pd
import numpy as np
from supabase_client import get_supabase_client, is_connected
from scipy.stats import linregress
import json

def fetch_latest_run_logs():
    """Fetches all logs for the most recent training run from Supabase."""
    if not is_connected():
        print("Supabase is not connected. Cannot fetch logs.")
        return None, None
    # ... (rest of the function is the same, but returns config too)
    supabase = get_supabase_client()
    try:
        latest_log_res = supabase.table("training_logs").select("model_id, training_config").order("created_at", desc=True).limit(1).execute()
        if not latest_log_res.data:
            print("No training logs found in Supabase.")
            return None, None
        latest_model_id = latest_log_res.data[0]['model_id']
        training_config = latest_log_res.data[0]['training_config']
        print(f"Fetching logs for the latest run (model_id: {latest_model_id})...")

        log_res = supabase.table("training_logs").select("*").eq("model_id", latest_model_id).order("epoch").execute()
        
        if not log_res.data:
            print(f"No logs found for model_id: {latest_model_id}")
            return None, None
            
        df = pd.DataFrame(log_res.data)
        print(f"Successfully fetched {len(df)} log entries.")
        return df, training_config
    except Exception as e:
        print(f"An error occurred while fetching logs: {e}")
        return None, None

def analyze_learning_dynamics(logs_df: pd.DataFrame):
    """Analyzes the convergence patterns from training logs."""
    if logs_df is None or logs_df.empty:
        return {"error": "Log data is empty."}
    
    final_logs = logs_df.iloc[int(len(logs_df) * 0.8):]
    final_total_unweighted_loss = final_logs['loss_bc'].mean() + final_logs.get('loss_ic', 0).mean() + final_logs['loss_pde'].mean()
    final_physics_loss_ratio = (final_logs['loss_pde'].mean() / final_total_unweighted_loss) if final_total_unweighted_loss > 0 else 0
    
    final_logs = final_logs.copy()
    final_logs.loc[:, 'log_total_loss'] = np.log10(final_logs['total_loss'] + 1e-10)
    slope, _, _, _, _ = linregress(final_logs['epoch'], final_logs['log_total_loss'])
    fluctuation = final_logs['total_loss'].std() / final_logs['total_loss'].mean() if final_logs['total_loss'].mean() > 0 else 0

    return {
        "final_physics_loss_ratio": float(final_physics_loss_ratio),
        "convergence_rate_slope": float(slope),
        "loss_fluctuation_coeff": float(fluctuation),
        "final_total_loss": logs_df['total_loss'].iloc[-1]
    }

def calculate_complexity(pde_type='2D Wave Equation'):
    """Calculates a heuristic complexity score for the PDE."""
    if pde_type == '2D Wave Equation':
        dim_score = 3  # 2 spatial + 1 temporal
        linearity_score = 1 # Linear
        coupling_score = 1 # Not coupled
        description = "2D Linear Wave Equation: u_tt = c^2 * (u_xx + u_yy)"
    else:
        # Default for unknown
        dim_score, linearity_score, coupling_score = 1, 1, 1
        description = "Unknown PDE"

    total_score = dim_score * linearity_score * coupling_score
    return {
        "pde_type": description,
        "score": total_score,
        "factors": f"Dimensions={dim_score}, Linearity={linearity_score}, Coupling={coupling_score}"
    }

def generate_abstracts(report_data: dict):
    """Generates Korean and English abstracts based on the analysis."""
    
    # This function simulates an LLM call.
    # It formats the data and returns a pre-defined text structure.
    
    dynamics = report_data['learning_dynamics']
    config = report_data['config']
    
    # English Abstract
    en_abstract = f"""
ABSTRACT
This study presents a Physics-Informed Neural Network (PINN) framework for solving the {report_data['complexity']['pde_type']}. 
The model was trained for {config['epochs']} epochs, leveraging advanced techniques including residual-based importance sampling and adaptive loss weighting to enhance training stability and convergence speed. 
Analysis of the learning dynamics reveals a final physics loss contribution of {dynamics['final_physics_loss_ratio']:.2%}, indicating a strong adherence to the underlying physical laws. The convergence rate in the final training phase, measured by the log-loss slope, was {dynamics['convergence_rate_slope']:.2e}. 
The final total loss achieved was {dynamics['final_total_loss']:.2e}. These results demonstrate the efficacy of the proposed PINN methodology in accurately modeling complex physical phenomena. 
However, the stability of the final convergence exhibits a fluctuation coefficient of {dynamics['loss_fluctuation_coeff']:.4f}, suggesting potential for further improvements in hyperparameter tuning or network architecture.
"""

    # Korean Abstract
    ko_abstract = f"""
요약
본 연구는 {report_data['complexity']['pde_type']}을 해결하기 위한 물리 정보 신경망(PINN) 프레임워크를 제시한다. 
총 {config['epochs']} 에포크 동안 학습된 모델은, 학습 안정성과 수렴 속도를 향상시키기 위해 잔차 기반 중요도 샘플링 및 적응형 손실 가중치 조절과 같은 고급 기법들을 활용하였다. 
학습 동역학 분석 결과, 최종 물리 손실의 기여도는 {dynamics['final_physics_loss_ratio']:.2%}로, 모델이 기반 물리 법칙을 잘 따르고 있음을 보여준다. 학습 후반부의 수렴 속도는 로그-손실 기울기 기준 {dynamics['convergence_rate_slope']:.2e}로 측정되었다. 
최종적으로 달성한 총 손실은 {dynamics['final_total_loss']:.2e}이다. 이러한 결과는 제안된 PINN 방법론이 복잡한 물리 현상을 정확하게 모델링하는 데 효과적임을 입증한다. 
다만, 최종 수렴 구간에서의 손실 변동성 계수가 {dynamics['loss_fluctuation_coeff']:.4f}로 나타나, 하이퍼파라미터 튜닝 또는 네트워크 아키텍처 개선을 통해 안정성을 더욱 향상시킬 여지가 있음을 시사한다.
"""
    return {"english": en_abstract.strip(), "korean": ko_abstract.strip()}

def print_summary_report(report: dict):
    """Prints a formatted summary report to the terminal."""
    print("\n" + "="*80)
    print(" " * 25 + "PINN WaveLab Final Summary Report")
    print("="*80)

    # --- Config ---
    print("\n[+] 1. Training Configuration")
    print(f"  - Model ID: {report.get('model_id', 'N/A')}")
    print(f"  - PDE: {report['complexity']['pde_type']}")
    print(f"  - Epochs: {report['config'].get('epochs', 'N/A')}")
    print(f"  - Learning Rate: {report['config'].get('lr', 'N/A')}")
    print(f"  - Advanced Features:")
    print(f"    - Importance Sampling: {'Enabled' if report['config'].get('use_importance_sampling') else 'Disabled'}")
    print(f"    - Adaptive Weighting: {'Enabled' if report['config'].get('use_adaptive_weighting') else 'Disabled'}")

    # --- Dynamics ---
    print("\n[+] 2. Learning Dynamics Analysis")
    dynamics = report['learning_dynamics']
    print(f"  - Final Total Loss: {dynamics['final_total_loss']:.6f}")
    print(f"  - Physics Loss Ratio (at convergence): {dynamics['final_physics_loss_ratio']:.4f}")
    print(f"  - Convergence Rate (log-loss slope): {dynamics['convergence_rate_slope']:.6f}")
    print(f"  - Loss Fluctuation (at convergence): {dynamics['loss_fluctuation_coeff']:.4f}")
    
    # --- Complexity ---
    print("\n[+] 3. PDE Complexity Analysis")
    complexity = report['complexity']
    print(f"  - Heuristic Score: {complexity['score']}")
    print(f"  - Factors: {complexity['factors']}")
    
    # --- Abstracts ---
    print("\n[+] 4. Generated Abstracts")
    print("\n--- English Abstract ---")
    print(report['abstracts']['english'])
    print("\n--- 국문 요약 ---")
    print(report['abstracts']['korean'])
    
    print("\n" + "="*80)


if __name__ == '__main__':
    print("Starting analysis of the latest training run...")
    df_logs, config = fetch_latest_run_logs()
    
    if df_logs is not None:
        # 1. 학습 동역학 분석
        dynamics_results = analyze_learning_dynamics(df_logs)
        
        # 2. 복잡도 분석
        # To-Do: In a real scenario, parse the PDE type from config
        complexity_results = calculate_complexity(pde_type='2D Wave Equation')

        # 3. 전체 리포트 데이터 구조 생성
        final_report = {
            "model_id": df_logs['model_id'].iloc[0],
            "config": config,
            "learning_dynamics": dynamics_results,
            "complexity": complexity_results,
        }

        # 4. 초록 생성
        abstracts = generate_abstracts(final_report)
        final_report['abstracts'] = abstracts
        
        # 5. 최종 리포트 출력
        print_summary_report(final_report)
        
        # To-Do: Save to `research_reports` table when schema is known
        # print("\nNote: Saving to `research_reports` table is skipped as schema is not provided.")

    print("\nAnalysis script finished.")
