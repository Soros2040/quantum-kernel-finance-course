# QuKerNet 金融迁移实验

本目录实现一套独立的研究协议：在股票历史截段上生成量子核对齐标签，训练线路评分器，筛选候选并微调，再以时间隔离的验证与季度样本评价。它是 QuKerNet 方法的金融迁移；运行规模与结果以私有运行记录为准。现有课程的四候选 KRR 是背景对照，不能替代这里的架构搜索验证。

金融数据处于 **D0 原始价格诊断**范围。完成程序运行不等于通过总回报数据准入、可执行策略回测或证明投资收益。本页对应历史本地版本，相关入口不提交云任务。新增云适配见独立的[cloud_v2模块](cloud_v2/README.md)，按其新版协议和授权执行。

## 方法与入口

| 文件 | 职责 |
|---|---|
| 本README | 公开的历史任务、预算和验收说明；原始运行契约保留在私有记录 |
| `data_pipeline.py` | 交易时间边界、成熟标签、训练窗内特征选择和缩放 |
| `quantum_engine.py` | PennyLane 精确态、保真度核、对齐和门参数微调 |
| `modeling.py` | 相容训练/交叉核谱映射、KRR、SVC和日期等权评价 |
| `run_search.py` | 两种候选选择方法、微调前后结果、经典对照、独立模型产物 |
| `finite_shots.py` | 冻结五候选的有限测量选择诊断 |
| `verify_engine.py` / `verify_data.py` | 数值与时间边界检查；通过情况以实际记录为准 |

8 比特、8 个已选择特征；角度输入为训练窗变换后的值乘以 π。一个 RX/RY/RZ 奇偶位布局与相邻 CNOT 开关组合重复1–5层，共枚举 **5,760种不同线路代码**。代码不同不保证核函数不同。随机排列后固定160条标签线路，其中128条训练评分器、32条评价评分器；随后4,096条代码作为新候选池。数值或科学退化候选记录原因，不另抽样补足。若原32条留出核与评分器训练核在最大绝对差 `1e-10` 内相同，则只从评分器评价中排除，仍保留原始审计记录。

评分器为128个ReLU隐藏单元、标量输出的MLP，监督值为10倍对齐，float64、SmoothL1、Adam学习率0.01、30轮、批量32。它预测线路质量，并非股票预测器。回归采用中心化核与中心化收益目标的对齐；分类校验使用作者式未中心化类别对齐和固定 `C=1` 的SVC。连续收益预测是明确的回归迁移。

QuKerNet方法对预测前5条计算完整对齐。随机搜索获得同160条初始标签计算和随机5条新线路的完整对齐，从165条已评价线路中选5条。160条标签的名义成本计入每种方法，即使物理计算被共享。每种方法的5条候选各有5个门替换配置、每配置30步、每步有放回抽16个样本、梯度下降步长0.2。相同深度和配置编号使用相同替换位置与小批次随机流。先按完整训练对齐选择每条线路的微调配置，再用金融验证集比较候选。

回归KRR以均方损失定义正则化，线性系统是 `K+nλI`，λ网格为 `[0.001,0.01,0.1]`。原始收益MSE及除以64行训练目标样本方差后的标准化MSE均按日期等权报告。RBF的γ为z空间中的 `[0.1,1,10]`，其中 `z=angles/π`，每个γ均保留λ搜索记录。背景对照还包括同特征线性模型、旧四个量子核、全成熟训练窗ridge与动量排序。全训练窗模型不是同标签数对照。

三季度为2022Q1、2023Q1、2024Q1，模型种子为20261109、20261110、20261111；数据抽样始终固定20261109。分类校验仅2024Q1第一种子。具体数据窗口、特征及标签成熟条件见本README和 `data_pipeline.py`。不读取2025年或以后记录。

## 运行方式

本次有记录的数值环境是 **Linux x86_64、Python 3.11.15、CPU**。PennyLane固定0.33.1、JAX/JAXlib固定0.4.20、Torch固定2.1.1+cpu；其他版本见 `environment.lock.txt`。新的机器需要自行保存安装与验收记录，不能把已有运行记录当成新环境已经通过的证据。

`bounded_execute.py` 使用 Linux 的 `resource`、CPU亲和性和进程组接口，正式资源受限流程应在 Linux 中运行；它不支持原生 Windows。WSL2需作为独立Linux环境配置并重新验收，本次未提供WSL2的运行证明。

### 1. 配置独立CPU环境

以下命令从**课程仓库根目录**执行。先把 `QK_ENV` 设置为自己选择的、课程目录以外的新虚拟环境目录。依赖安装会访问软件包源；Notebook及正式实验本身不需要联网。CPU版Torch先从其官方CPU wheel源安装，随后安装本模块锁定的依赖。`pip`和`setuptools`另行固定，因为普通 `pip freeze` 清单不包含全部构建工具。

```bash
: "${QK_ENV:?Set QK_ENV to a new environment directory outside the course}"
python3.11 -m venv "$QK_ENV"
"$QK_ENV/bin/python" -m pip install pip==25.2 setuptools==68.2.2
"$QK_ENV/bin/python" -m pip install torch==2.1.1+cpu --index-url https://download.pytorch.org/whl/cpu
"$QK_ENV/bin/python" -m pip install -r experiments/qukernet_finance/environment.lock.txt
"$QK_ENV/bin/python" -m pip check
```

以上版本和CPU wheel源来自本次实际环境记录；本次审阅没有在另一台空白机器上重新安装。锁文件不是跨操作系统的安装保证，也不包含Notebook前端。无需安装Jupyter即可按下面的方式顺序执行解析Notebook的原代码单元。

### 2. 无金融数据时运行解析Notebook

保持工作目录为课程仓库根目录，沿用上述 `QK_ENV`。以下命令只读取Notebook，使用同一个命名空间顺序执行其7个代码单元，结果输出到终端；不改写Notebook或保存其输出。示例使用固定种子的解析输入，默认 `PRIVATE_OUTPUT=None`。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 "$QK_ENV/bin/python" -B - <<'PY'
import json
from pathlib import Path
notebook = Path("experiments/qukernet_finance/minimal_analytic_demo.ipynb")
document = json.loads(notebook.read_text(encoding="utf-8"))
namespace = {"__name__": "__notebook_demo__"}
for index, cell in enumerate(document["cells"]):
    if cell["cell_type"] == "code":
        exec(compile("".join(cell["source"]), f"notebook_cell_{index}", "exec"), namespace)
PY
```

也可以在自己已有的Notebook前端中选择该Python环境并顺序运行；前端和内核绑定需自行配置，不能假设锁文件已安装Jupyter或 `ipykernel`。本次已在Python 3.11.15下逐cell执行原Notebook的7个代码单元，全部通过；验收记录为 `actual_notebook_check.json`，Notebook及其三个源模块的哈希与公开文件一致，源文件未被改写。Notebook中的 `language_info.version=3.10` 是保留的旧元数据，不代表实际执行版本。公开Notebook保留空输出，不包含任何历史证券结果；新机器仍需重新验收。

### 3. 有合法数据时准备两个本地CSV

取得使用授权后，将输入冻结为本地只读快照；本模块不提供行情下载器。不要把数据、训练后参数或运行产物提交到公开课程目录。证券集合必须事先固定，记录其选择时间和依据；不能根据待评价季度的完整性、收益或预测表现重新选择证券。不同合法数据构成方法复现，不会自动复现本次私有快照的数值结果。

`INPUT_CSV` 的最小字段如下，UTF-8编码，每个“证券—交易日”最多一行：

| 字段 | 格式与用途 |
|---|---|
| `symbol` | 非空字符串；整个快照恰好20个不同标识。不要把前导零丢失。 |
| `date` | 不带时区的日日期，例如ISO `YYYY-MM-DD`；不得包含日内时分秒。 |
| `open` | 同一原始价格口径的日开盘价，用于标签。 |
| `close` | 同一原始价格口径的日收盘价，用于动量和波动特征。 |
| `volume` | 成交量；各日期采用一致单位，用于相对成交量。 |

`high`、`low`、`prev_close`、`amount`等原始字段可以保留，但当前12特征及标签不使用它们。`open/close/volume`无法转为数值或非正时作为缺失处理；缺口不前填、不后填，也不压缩交易日。原始价诊断不能混入单边复权的开盘/收盘序列，更不能自动解释为公司行动调整后的总回报。

`CALENDAR_CSV` 至少有一个 `date` 列，包含对应市场的完整交易日集合，同样为不带时区的日日期且无重复。股票行的日期必须都存在于该日历中。股票停牌或缺行情的日期仍应保留在市场日历中；不能用某一只股票的实际有记录日期替代市场日历。

本冻结协议只接受2019–2024记录，输入中出现2018及以前或2025及以后日期会直接拒绝。其他直接拒绝条件包括缺字段、重复证券日期、重复日历日期、空日期/空证券标识、时区或日内时间、证券数不等于20、面板日期不在日历中。所选季度还必须有足够的前252个市场日及特征所需历史，能取得规定的64行训练样本和两个完整20证券验证截面，否则不能临时补齐或换证券继续。

预测以交易日历索引为准：在日期 `t` 使用截至 `t` 的特征，连续收益标签为 `open(t+6)/open(t+1)-1`，分类标签为该收益严格大于0时取1、否则取0。`label_end=t+6`。训练标签必须在验证区间开始前成熟，验证标签必须在季度开始前成熟，季度评价标签也必须在该季度内结束。窗口为前252市场日的168日训练、84日验证；全部缩放和mRMR只拟合64行受控训练样本。具体选择规则与缺失覆盖记录写入窗口审计。

### 4. 检查、预热和正式运行

下面从模块目录执行。先设置 `INPUT_CSV`、`CALENDAR_CSV` 为已准入本地快照路径，`PRIVATE_WORK` 为课程目录及输入目录之外的新私有工作目录。每次启动前将 `REMAINING_SECONDS` 设置为本次72小时总窗口的**剩余秒数**；它不是每次重启的新额度。`PRIVATE_WORK`及其缓存必须控制在10 GiB以内。

```bash
cd experiments/qukernet_finance
: "${INPUT_CSV:?Set the admitted local panel CSV}"
: "${CALENDAR_CSV:?Set the admitted local market calendar CSV}"
: "${PRIVATE_WORK:?Set a private local work directory outside the course and inputs}"
: "${REMAINING_SECONDS:?Set the remaining seconds in this execution window}"
"$QK_ENV/bin/python" -B verify_data.py --input "$INPUT_CSV" --calendar "$CALENDAR_CSV" --output "$PRIVATE_WORK/data_checks"
"$QK_ENV/bin/python" -B data_pipeline.py --input "$INPUT_CSV" --calendar "$CALENDAR_CSV" --output "$PRIVATE_WORK/windows"
"$QK_ENV/bin/python" -B bounded_execute.py --output "$PRIVATE_WORK/runs" --name preflight --seconds "$REMAINING_SECONDS" --cache-root "$PRIVATE_WORK" -- "$QK_ENV/bin/python" -B run_search.py --input "$INPUT_CSV" --calendar "$CALENDAR_CSV" --output "$PRIVATE_WORK/runs/preflight" --preflight
```

检查数据验收与预热输出后，再更新 `REMAINING_SECONDS`，启动正式流程。下列命令是后续独立步骤，不应在预热失败后自动接续执行。

```bash
"$QK_ENV/bin/python" -B bounded_execute.py --output "$PRIVATE_WORK/runs" --name full --seconds "$REMAINING_SECONDS" --cache-root "$PRIVATE_WORK" -- "$QK_ENV/bin/python" -B run_search.py --input "$INPUT_CSV" --calendar "$CALENDAR_CSV" --output "$PRIVATE_WORK/runs/full"
```

只有正式运行生成完整的五候选 `finite_input.json` 后才运行有限测量诊断。其固定位置是正式输出下的 `regression/2024Q1/seed_20261109/finite_input.json`；应先核对其状态为 `complete`，并再次更新剩余时间。

```bash
"$QK_ENV/bin/python" -B bounded_execute.py --output "$PRIVATE_WORK/runs" --name finite --seconds "$REMAINING_SECONDS" --cache-root "$PRIVATE_WORK" -- "$QK_ENV/bin/python" -B finite_shots.py --source "$PRIVATE_WORK/runs/full/regression/2024Q1/seed_20261109/finite_input.json" --output "$PRIVATE_WORK/runs/finite"
```

恢复完整缓存时给 `run_search.py` 增加 `--resume`，正式输出目录保持一致，同时为外层执行器使用一个新的 `--name`，保留原执行日志。失败、超时或残留运行中单元的自动重跑会被拒绝；不要手动改写它们为成功。代码或协议变更使用新的私有输出目录。

小试仅计算8条标签线路和一个配置的完整30步微调，应使用与正式运行不同的输出目录。正式运行可用 `--quarters`、`--seeds` 限定已批准组合，实际组合写入清单。恢复仅接受输入、代码、环境及协议哈希一致且完整校验通过的记录。状态为运行中、失败或超时的任务不会自动重新执行。参数变更需要新输出目录。

量子任务通过 `spawn` 持久子进程串行执行，单核任务300秒、单配置1,800秒；不在已初始化JAX后派生fork进程。外层资源执行器须约束进程树不超过8线程、32 GiB RSS、10 GiB私有缓存及72小时总窗口。72小时是停止上限，不是完成承诺。失败后停止该阶段扩规模，保留已完成和失败记录。

执行器记录中的 `location="Wuying CPU"` 是本次运行批次的固定标识，不是自动硬件探测结果。外部复用应另记实际机器、系统和资源信息，不能把该字符串当作自己运行所在机器的证据。

模型产物保存布局、替换位置、训练后参数、输入特征顺序、预处理、训练角度、目标统计量、下游模型、预测及代码哈希。KRR经过JSON重载后的预测要满足 `atol=rtol=1e-8`。训练核的PSD修正和验证/季度交叉核使用同一训练谱延拓，随后再中心化和缩放。

## 有限测量诊断

固定2024Q1第一种子经过精确搜索产生的五条QuKerNet微调候选，只研究它们的**最后一次选择**。候选池本身使用精确信息构造，这一范围不能隐藏。任何候选缺失或失效都不能临时换入另一条线路补足五条。

每个候选使用64训练、40验证样本，λ固定0.1，没有校准阶段。训练上三角和验证交叉项以独立二项采样模拟估计；理想归一化对角为1不计shots。不同候选没有物理读数共享。`S=32/128/512`各20次，总名义预算是：

```text
5 × [64×63/2 + 40×64] × (32+128+512) × 20 = 307,507,200
```

这是本地概率采样所对应的虚拟执行预算，真实云shots为0。每次根据采样后的训练核重新做谱修正，并相容映射交叉核；不能复用精确核的特征投影。顶层 `select_candidate` 只接测量矩阵、标签、日期和λ，不接精确损失或精确特征。全部选择记录保存后才计算精确参照，并由另一个顶层函数 `evaluate_choice` 进行事后评估。

报告包括选择频数、重复间一致性、候选差分的标准差与符号翻转、事后遗憾、δ为0.01/0.02/0.05的失选率及Wilson区间。δ的单位为标准化预测MSE，不能解释为收益或金钱损失。部分候选失败和全候选失败均保留；不能以零损失替代失败。区间只描述这个固定候选池、固定数据上的重复采样，不证明未见季度泛化、器件噪声鲁棒性或物理独立同分布。

## 上游与可核验边界

公开实现依据 [QuKerNet作者仓库](https://github.com/iamconglei-ctrl/QuKerNet)，固定提交 `c9948606145eb85c2bf53f27acebaf5a071a3fa7`，其许可为 [MIT](https://github.com/iamconglei-ctrl/QuKerNet/blob/c9948606145eb85c2bf53f27acebaf5a071a3fa7/LICENSE)。复用上游代码时保留相应版权与许可通知。股票数据许可单独核验。

本实现明确绑定微调掩码和参数，并以验证集选择模型；作者公开示例只保存参数、测试重新抽掩码，且其脚本规模不是论文全部实验配置。迁移结果不能冒充原论文数值复现。回归、时间划分、谱映射和预算记录均属于本研究显式定义的协议。

8比特的精确核可由至多65,536维Hermitian密度特征表示；4比特一般复态需要至多256维。旧136维实对称特征只适用于实态。数值等价与搜索性能均不能单独证明量子计算优势。精确状态缓存的实际成本、名义成对核调用、自动微分和实测时间须分别报告；当前共享暖启动结果的两条流水线不构成独立的等耗时实验。

## 独立环境与解析示例

实际依赖版本见 `environment.lock.txt`。本模块使用 PennyLane `default.qubit` 的双精度精确态与 autograd/backprop；JAX 在环境中保留，但这条执行路径不使用 JIT。PennyLane 0.33.1 需要兼容的 `autoray==0.6.7`、`autograd==1.6.2` 及 `setuptools==68.2.2`；在新的虚拟环境中安装，不改变其他课程环境。PyTorch 为 CPU 版2.1.1。

`minimal_analytic_demo.ipynb` 是不含证券数据的最小示例，展示编码、精确核、中心化对齐、KRR和有限测量的最后选择。运行时从课程仓库内启动。Notebook默认不写文件、不联网、不提交云任务。完整金融运行应通过 `bounded_execute.py` 包装，指定明确的私有输出目录；命令中的 `--seconds` 应取本轮72小时窗口的剩余时间，不能每次重启重新获得72小时。

`benchmark20.py` 是20比特解析输入的核块性能检查，和金融架构搜索分别计量。`reproduction_differences.csv` 记录论文、固定版本代码和本次设置之间的区别。任何运行结果均须以独立保存的验收记录为依据。
