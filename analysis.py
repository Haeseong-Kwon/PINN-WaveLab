import pandas as pd
import numpy as np
from supabase_client import get_supabase_client, is_connected
from scipy.stats import linregress

def fetch_latest_run_logs():
    """
    Fetches all logs for the most recent training run from Supabase.
    A "run" is identified by having the same model_id.
    """
    if not is_connected():
        print("Supabase is not connected. Cannot fetch logs.")
        return None

    supabase = get_supabase_client()
    
    # 1. 가장 최근에 기록된 로그의 model_id를 찾습니다.
    try:
        latest_log_res = supabase.table("training_logs").select("model_id").order("created_at", desc=True).limit(1).execute()
        if not latest_log_res.data:
            print("No training logs found in Supabase.")
            return None
        latest_model_id = latest_log_res.data[0]['model_id']
        print(f"Fetching logs for the latest run (model_id: {latest_model_id})...")

        # 2. 해당 model_id를 가진 모든 로그를 가져옵니다.
        log_res = supabase.table("training_logs").select("*").eq("model_id", latest_model_id).order("epoch").execute()
        
        if not log_res.data:
            print(f"No logs found for model_id: {latest_model_id}")
            return None
            
        df = pd.DataFrame(log_res.data)
        print(f"Successfully fetched {len(df)} log entries.")
        return df

    except Exception as e:
        print(f"An error occurred while fetching logs: {e}")
        return None


def analyze_learning_dynamics(logs_df: pd.DataFrame):
    """
    Analyzes the convergence patterns from training logs.
    """
    if logs_df is None or logs_df.empty:
        return {
            "error": "Log data is empty or invalid.",
            "final_loss_ratio": None,
            "convergence_rate": None,
            "loss_fluctuation": None
        }

    # 마지막 20%의 데이터를 사용하여 수렴 특성 분석
    final_logs = logs_df.iloc[int(len(logs_df) * 0.8):]

    # 1. 최종 손실 비율 (물리 손실이 전체 손실에서 차지하는 비중)
    # total_loss는 가중치가 적용된 합이므로, 가중치 없는 손실의 합을 기준으로 계산
    final_total_unweighted_loss = final_logs['loss_bc'].mean() + final_logs['loss_ic'].mean() + final_logs['loss_pde'].mean()
    final_physics_loss_ratio = (final_logs['loss_pde'].mean() / final_total_unweighted_loss) if final_total_unweighted_loss > 0 else 0
    
    # 2. 수렴 속도 (마지막 20% 구간의 로그-손실 그래프 기울기)
    # 로그 스케일에서 선형 회귀를 사용하여 기울기 계산
    final_logs['log_total_loss'] = np.log10(final_logs['total_loss'] + 1e-10)
    slope, _, _, _, _ = linregress(final_logs['epoch'], final_logs['log_total_loss'])

    # 3. 손실 변동성 (마지막 20% 구간의 손실 값의 변동 계수)
    fluctuation = final_logs['total_loss'].std() / final_logs['total_loss'].mean() if final_logs['total_loss'].mean() > 0 else 0

    return {
        "final_physics_loss_ratio": float(final_physics_loss_ratio),
        "convergence_rate_slope": float(slope),
        "loss_fluctuation_coeff": float(fluctuation)
    }

if __name__ == '__main__':
    print("Starting analysis of the latest training run...")
    df_logs = fetch_latest_run_logs()
    
    if df_logs is not None:
        dynamics_results = analyze_learning_dynamics(df_logs)
        
        print("
--- Learning Dynamics Analysis ---")
        if "error" in dynamics_results:
            print(f"Analysis failed: {dynamics_results['error']}")
        else:
            print(f"  - Final Physics Loss Ratio: {dynamics_results['final_physics_loss_ratio']:.4f}")
            print(f"    (Proportion of physics loss in the final unweighted total loss)")
            print(f"  - Convergence Rate (Slope): {dynamics_results['convergence_rate_slope']:.6f}")
            print(f"    (Slope of log-loss vs. epochs in the final 20% of training)")
            print(f"  - Loss Fluctuation Coeff: {dynamics_results['loss_fluctuation_coeff']:.4f}")
            print(f"    (Coefficient of variation of total loss in the final 20%)")

    print("
Analysis script finished.")
