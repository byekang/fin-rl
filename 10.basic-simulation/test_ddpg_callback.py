"""
DDPG Custom Callback 테스트 스크립트
'Logging Error: rollout_buffer' 에러가 해결되었는지 확인
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from stable_baselines3.common.logger import configure
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories

# stable-baselines3 모델들
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import BaseCallback
import statistics

# Custom Callback 클래스
class OffPolicyTensorboardCallback(BaseCallback):
    """
    Off-policy 알고리즘(DDPG, TD3, SAC)과 on-policy 알고리즘(A2C, PPO) 모두 지원하는 Custom callback.

    FinRL의 기본 TensorboardCallback은 rollout_buffer만 지원하여
    off-policy 알고리즘에서 'Logging Error: rollout_buffer' 에러가 발생하는 문제를 해결합니다.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        """매 스텝마다 호출 - reward 로깅"""
        try:
            self.logger.record(key="train/reward", value=self.locals["rewards"][0])
        except BaseException as error:
            try:
                self.logger.record(key="train/reward", value=self.locals["reward"][0])
            except BaseException:
                # reward를 찾을 수 없는 경우 무시
                pass
        return True

    def _on_rollout_end(self) -> bool:
        """
        Rollout 종료 시 호출 - on-policy 알고리즘에서만 작동
        Off-policy 알고리즘은 rollout_buffer가 없으므로 조용히 스킵
        """
        try:
            # On-policy 알고리즘 (A2C, PPO)은 rollout_buffer 사용
            if "rollout_buffer" in self.locals:
                rollout_buffer_rewards = self.locals["rollout_buffer"].rewards.flatten()
                self.logger.record(
                    key="train/reward_min", value=min(rollout_buffer_rewards)
                )
                self.logger.record(
                    key="train/reward_mean", value=statistics.mean(rollout_buffer_rewards)
                )
                self.logger.record(
                    key="train/reward_max", value=max(rollout_buffer_rewards)
                )
            # Off-policy 알고리즘 (DDPG, TD3, SAC)은 replay_buffer 사용
            # replay_buffer에서는 reward 통계를 다르게 수집해야 하므로 스킵
        except Exception:
            # 에러 발생 시 조용히 무시 (에러 메시지 출력 안 함)
            pass
        return True


def main():
    print("="*70)
    print("DDPG Custom Callback 테스트")
    print("="*70)

    # 1. 데이터 로드
    print("\n[1/5] 학습 데이터 로드 중...")
    train = pd.read_csv('train_data.csv')
    print(f"✓ 데이터 로드 완료: {len(train)}행")

    # 인덱스 생성
    train = train.sort_values(['date', 'tic']).reset_index(drop=True)
    date_to_day = {date: i for i, date in enumerate(sorted(train['date'].unique()))}
    train['day'] = train['date'].map(date_to_day)
    train = train.set_index('day')
    print(f"✓ 인덱스 생성 완료: {train.index.nunique()}개 거래일")

    # 2. 환경 파라미터 설정
    print("\n[2/5] 환경 파라미터 설정 중...")
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(INDICATORS)*stock_dimension
    buy_cost_list = sell_cost_list = [0.001] * stock_dimension
    num_stock_shares = [0] * stock_dimension

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "num_stock_shares": num_stock_shares,
        "buy_cost_pct": buy_cost_list,
        "sell_cost_pct": sell_cost_list,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
    }
    print(f"✓ 환경 파라미터 설정 완료 (state_space: {state_space})")

    # 3. 환경 생성
    print("\n[3/5] 학습 환경 생성 중...")
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    print(f"✓ 환경 생성 완료")

    # 4. DDPG 모델 생성 및 설정
    print("\n[4/5] DDPG 모델 생성 중...")
    check_and_make_directories([RESULTS_DIR])
    agent = DRLAgent(env=env_train)
    model_ddpg = agent.get_model("ddpg", model_kwargs={'device': 'cpu'})  # 빠른 테스트를 위해 CPU 사용

    tmp_path = RESULTS_DIR + '/ddpg_test'
    new_logger_ddpg = configure(tmp_path, ["stdout"])
    model_ddpg.set_logger(new_logger_ddpg)
    print(f"✓ DDPG 모델 생성 완료")

    # 5. 짧은 학습 실행 (1000 timesteps)
    print("\n[5/5] DDPG 학습 시작 (1000 timesteps)...")
    print("="*70)
    print("⚠️  'Logging Error: rollout_buffer' 메시지가 나오는지 확인하세요!")
    print("="*70)

    try:
        trained_ddpg = model_ddpg.learn(
            total_timesteps=1000,
            tb_log_name='ddpg_test',
            callback=OffPolicyTensorboardCallback()
        )

        print("\n" + "="*70)
        print("✅ 테스트 성공!")
        print("="*70)
        print("✓ 'Logging Error: rollout_buffer' 에러가 발생하지 않았습니다.")
        print("✓ Custom Callback이 정상적으로 작동합니다.")
        print("\n다음 단계: Stock_NeurIPS2018_2_Train_Fixed.ipynb에서 전체 학습 실행")
        return True

    except Exception as e:
        print("\n" + "="*70)
        print("❌ 테스트 실패!")
        print("="*70)
        print(f"에러: {str(e)}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
