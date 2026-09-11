#!/bin/bash
# ==============================================================================
# Dual Arm Imitation Learning (Visuomotor): Full Pipeline Script 
# ==============================================================================

set -e  # 에러 발생 시 즉시 종료

# 설정 (자율적으로 수정 가능)
ALGO="diffusion"
NUM_DEMOS=2000   # Handover는 난이도가 높으므로 많은 데모 권장
NUM_ENVS=16      # 병렬 환경 개수 (메모리에 맞게 조절)
EPOCHS=1500      # 학습 에폭
RUN_EVAL=true    # 병렬 평가(eval_parallel.py)를 실행할지 여부 (true/false)
NUM_EVAL_EPISODES=100 # 평가할 에피소드 수
PYTHON_EXEC="~/isaac_lab/bin/python"

echo "============================================================================="
echo "🚀 Visuomotor 파이프라인 시작 (Data Collection -> Train -> Eval)"
echo "알고리즘: $ALGO | 데모 개수: $NUM_DEMOS | 학습 에폭: $EPOCHS"
echo "============================================================================="

# 1. 기존 결과 파일 초기화
> pipeline_summary.txt

# 2. 데이터 수집
echo -e "\n[1/3] 🎥 $NUM_DEMOS 개의 Visuomotor 데모 데이터 수집 중... (병렬 $NUM_ENVS 환경)"
eval $PYTHON_EXEC scripts/generate_scripted_demos_parallel.py --num_demos $NUM_DEMOS --num_envs $NUM_ENVS --headless
echo "✅ 데이터 수집 완료."

# 3. 모델 학습
echo -e "\n[2/3] 🧠 $ALGO 모델 학습 중 ($EPOCHS Epochs)..."
eval $PYTHON_EXEC scripts/train.py --algo $ALGO --epochs $EPOCHS
echo "✅ 학습 완료."

# 4. 병렬 평가 (eval_parallel.py)
if [ "$RUN_EVAL" = true ] || [ "$RUN_EVAL" = "true" ]; then
    echo -e "\n[3/3] 🚀 병렬 평가 (eval_parallel.py) $NUM_EVAL_EPISODES 에피소드 진행 중..."
    PAR_LOG="results_eval_parallel.txt"
    eval $PYTHON_EXEC scripts/eval_parallel.py --algo $ALGO --num_episodes $NUM_EVAL_EPISODES --num_envs $NUM_ENVS --headless > $PAR_LOG 2>&1 || true
    PAR_SUCCESS=$(grep "Success Rate" $PAR_LOG | tail -n 1 | xargs)
    echo "✅ 병렬 평가 완료. ($PAR_SUCCESS)"
    echo "[eval_parallel.py] $PAR_SUCCESS" >> pipeline_summary.txt
else
    echo -e "\n[3/3] ⏭️ 병렬 평가 (eval_parallel.py) 건너뜀 (RUN_EVAL=false)"
    echo "[eval_parallel.py] Skipped" >> pipeline_summary.txt
fi

echo "============================================================================="
echo "🎉 Visuomotor 파이프라인이 종료되었습니다! 최종 결과 요약:"
cat pipeline_summary.txt
echo "============================================================================="
