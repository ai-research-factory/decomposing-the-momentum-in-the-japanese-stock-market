# Decomposing the Momentum in the Japanese Stock Market

## Project ID
proj_16402310

## Taxonomy
StatArb, ResidualFactors

## Current Cycle
3

## Objective
Implement, validate, and iteratively improve the paper's approach with production-quality standards.


## Design Brief
### Problem
This paper addresses the nature of the momentum effect in the Japanese stock market, a phenomenon where past winning stocks tend to continue winning and past losers continue losing. The core problem is to determine whether this effect is driven by the momentum of entire industries or by the performance of individual stocks independent of their industry. The paper hypothesizes that industry momentum is the primary driver and that it is negatively correlated with stock-specific momentum. The goal is to decompose the overall momentum factor into these two components, build trading strategies based on each, and evaluate their respective contributions to profitability and risk-adjusted returns over a long historical period (1980-2022).

### Datasets
Japanese stock prices (TOPIX constituents) from 1980-2022. Source: stooq (via pandas-datareader). Fallback: Generate synthetic data with multiple stocks and industries, exhibiting trend and mean-reversion properties.
- Industry classifications for Japanese stocks. Source: A manually curated CSV file mapping stock tickers to industry sectors (e.g., GICS or TOPIX-17). Fallback: Assign stocks to a small number of synthetic industries randomly.

### Targets
The primary target is to maximize the risk-adjusted return (Sharpe Ratio) of portfolios constructed from decomposed momentum factors.
- A secondary target is to accurately reproduce the paper's key findings: 1) Industry momentum is the dominant component of total momentum. 2) There is a negative correlation between industry momentum and stock-specific momentum.

### Model
The model is a statistical factor decomposition approach, not a machine learning model. It follows these steps: 1) Calculate a standard momentum score for each stock (e.g., 12-month past return). 2) Calculate an industry momentum score, typically as the average momentum of stocks within each industry. 3) Decompose the stock's total return into an industry component and a stock-specific (residual) component, likely via a single-factor regression: StockReturn_t = α + β * IndustryReturn_t + ε_t. 4) The stock-specific momentum is then calculated based on the momentum of these residuals (ε). 5) Portfolios are formed by going long on assets with high scores and short on assets with low scores for each of the three momentum factors (total, industry, stock-specific).

### Training
The 'training' in this context is the calculation of factor values and regression betas over a lookback period. The system uses a walk-forward validation approach. For each period (e.g., a year), a lookback window (e.g., the preceding 12-24 months) is used to calculate momentum scores and regression parameters. These parameters are then used to form a portfolio which is held for the duration of the forward period (e.g., 1 month). This process is repeated, rolling the window forward through time, ensuring there is no look-ahead bias. There is no traditional model training with epochs or gradient descent.

### Evaluation
The primary evaluation method is a multi-split walk-forward backtest over the entire 1980-2022 period. The performance of three long-short portfolios (Total Momentum, Industry Momentum, Stock-Specific Momentum) will be compared. Key metrics include: Cumulative Annual Growth Rate (CAGR), Sharpe Ratio, Maximum Drawdown, and portfolio turnover. The evaluation will also explicitly calculate and report the time-series correlation between the industry and stock-specific momentum factors to validate one of the paper's core findings. Performance will be reported both gross and net of transaction costs.


## データ取得方法（共通データ基盤）

**合成データの自作は禁止。以下のARF Data APIからデータを取得すること。**

### ARF Data API
```bash
# OHLCV取得 (CSV形式)
curl -o data/aapl_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y"
curl -o data/btc_1h.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=BTC/USDT&interval=1h&period=1y"
curl -o data/nikkei_1d.csv "https://ai.1s.xyz/api/data/ohlcv?ticker=^N225&interval=1d&period=10y"

# JSON形式
curl "https://ai.1s.xyz/api/data/ohlcv?ticker=AAPL&interval=1d&period=5y&format=json"

# 利用可能なティッカー一覧
curl "https://ai.1s.xyz/api/data/tickers"
```

### Pythonからの利用
```python
import pandas as pd
API = "https://ai.1s.xyz/api/data/ohlcv"
df = pd.read_csv(f"{API}?ticker=AAPL&interval=1d&period=5y")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.set_index("timestamp")
```

### ルール
- **リポジトリにデータファイルをcommitしない** (.gitignoreに追加)
- 初回取得はAPI経由、以後はローカルキャッシュを使う
- data/ディレクトリは.gitignoreに含めること



## ★ 今回のタスク (Cycle 3)


### Phase 3: 評価フレームワークの実装

**ゴール**: トータルモメンタム戦略を評価するための基本的なウォークフォワード検証を実装する。

**具体的な作業指示**:
1. `src/evaluation.py`を新規作成し、トータルモメンタム戦略のウォークフォワード評価フレームワークを実装する。
2. `src/backtest.py`の`WalkForwardValidator`, `calculate_costs`, `compute_metrics`, `generate_metrics_json`を活用する。
3. パネルデータからモメンタムスコアを計算し、ロング・ショートポートフォリオを構築する。
4. 各ウォークフォワード窓でのリターンを計算し、取引コストを考慮した評価を行う。
5. `tests/test_evaluation.py`にユニットテストを作成する。

**期待される出力ファイル**:
- src/evaluation.py（新規）
- tests/test_evaluation.py（新規）
- reports/cycle_3/metrics.json
- reports/cycle_3/technical_findings.md

**受入基準**:
- ウォークフォワード検証が正しく動作し、各窓でのOOSメトリクスが計算される。
- `tests/test_evaluation.py`のユニットテストが全て成功する。
- 既存テスト（`test_decomposition.py` 10件、`test_data_pipeline.py` 16件）も引き続き成功する。
- `reports/cycle_3/metrics.json`がARF標準スキーマに準拠する。










## 全体Phase計画 (参考)

✓ Phase 1: コア分解アルゴリズムの実装 — 合成データ上で株価の勢いを業界要因と個別要因に分解するコアロジックを実装する。
✓ Phase 2: 実データパイプラインの構築 — 実際の日本株価データを取得し、前処理を行い、保存するパイプラインを構築する。
→ Phase 3: 評価フレームワークの実装 — トータルモメンタム戦略を評価するための基本的なウォークフォワード検証を実装する。
  Phase 4: 分解戦略のバックテスト — 分解アルゴリズムを評価フレームワークに統合し、3つの戦略（全体、業界、個別）をバックテストする。
  Phase 5: 取引コストの計算 — 取引コストモデルを実装し、グロスリターンとネットリターンを比較評価する。
  Phase 6: ハイパーパラメータ最適化 — モメンタムのルックバック期間や保有期間などの主要パラメータを最適化する。
  Phase 7: ロバスト性検証 — 最適化されたパラメータを用いて全期間でのバックテストを実行し、戦略の頑健性を確認する。
  Phase 8: 相関分析 — 業界モメンタムと個別株モメンタムの間に負の相関があるという論文の発見を再現する。
  Phase 9: 期間別分析 — 戦略のパフォーマンスを年代別（1980年代、90年代など）に分析し、パフォーマンスの一貫性を評価する。
  Phase 10: 業界寄与度分析 — 業界モメンタム戦略の損益に最も貢献した業界を特定する。
  Phase 11: テクニカルレポート生成 — すべての分析結果をまとめた包括的なテクニカルレポートを自動生成する。
  Phase 12: 最終仕上げとエグゼクティブサマリー — テストカバレッジを向上させ、非技術者向けのプロジェクトサマリーを作成する。


## 評価原則
- **主指標**: Sharpe ratio (net of costs) on out-of-sample data
- **Walk-forward必須**: 単一のtrain/test splitでの最終評価は不可
- **コスト必須**: 全メトリクスは取引コスト込みであること
- **安定性**: Walk-forward窓の正の割合を報告
- **ベースライン必須**: 必ずナイーブ戦略と比較

## 禁止事項
- 未来情報を特徴量やシグナルに使わない
- 全サンプル統計でスケーリングしない (train-onlyで)
- テストセットでハイパーパラメータを調整しない
- コストなしのgross PnLだけで判断しない
- 時系列データにランダムなtrain/test splitを使わない
- APIキーやクレデンシャルをコミットしない
- **新しい `scripts/run_cycle_N.py` や `scripts/experiment_cycleN.py` を作成しない。既存の `src/` 内ファイルを修正・拡張すること**
- **合成データを自作しない。必ずARF Data APIからデータを取得すること**
- **「★ 今回のタスク」以外のPhaseの作業をしない。1サイクル=1Phase**

## Git / ファイル管理ルール
- **データファイル(.csv, .parquet, .h5, .pkl, .npy)は絶対にgit addしない**
- `__pycache__/`, `.pytest_cache/`, `*.pyc` がリポジトリに入っていたら `git rm --cached` で削除
- `git add -A` や `git add .` は使わない。追加するファイルを明示的に指定する
- `.gitignore` を変更しない（スキャフォールドで設定済み）
- データは `data/` ディレクトリに置く（.gitignore済み）
- 学習済みモデルは `models/` ディレクトリに置く（.gitignore済み）

## 出力ファイル
以下のファイルを保存してから完了すること:
- `reports/cycle_2/metrics.json` — 下記スキーマに従う（必須）
- `reports/cycle_2/technical_findings.md` — 実装内容、結果、観察事項

### metrics.json 必須スキーマ
```json
{
  "sharpeRatio": 0.0,
  "annualReturn": 0.0,
  "maxDrawdown": 0.0,
  "hitRate": 0.0,
  "totalTrades": 0,
  "transactionCosts": { "feeBps": 10, "slippageBps": 5, "netSharpe": 0.0 },
  "walkForward": { "windows": 0, "positiveWindows": 0, "avgOosSharpe": 0.0 },
  "customMetrics": {}
}
```
- 全フィールドを埋めること。Phase 1-2で未実装のメトリクスは0.0/0で可。
- `customMetrics`に論文固有の追加メトリクスを自由に追加してよい。
- `docs/open_questions.md` — 未解決の疑問と仮定
- `README.md` — 今回のサイクルで変わった内容を反映して更新（セットアップ手順、主要な結果、使い方など）
- `docs/open_questions.md` に以下も記録:
  - ARF Data APIで問題が発生した場合（エラー、データ不足、期間の短さ等）
  - CLAUDE.mdの指示で不明確な点や矛盾がある場合
  - 環境やツールの制約で作業が完了できなかった場合

## 標準バックテストフレームワーク

`src/backtest.py` に以下が提供済み。ゼロから書かず、これを活用すること:
- `WalkForwardValidator` — Walk-forward OOS検証のtrain/test split生成
- `calculate_costs()` — ポジション変更に基づく取引コスト計算
- `compute_metrics()` — Sharpe, 年率リターン, MaxDD, Hit rate算出
- `generate_metrics_json()` — ARF標準のmetrics.json生成

```python
from src.backtest import WalkForwardValidator, BacktestConfig, calculate_costs, compute_metrics, generate_metrics_json
```

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli run-experiment --config configs/default.yaml
```

Commit all changes with descriptive messages.
