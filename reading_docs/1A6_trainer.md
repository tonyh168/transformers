# Trainer 主循环精读

> 阅读目标：把 `Trainer` 这 4477 行拆成「一句话能说清的流程」——
> `__init__` 准备零件 → `train()` 进入循环 → 内层逐 micro-batch forward/backward →
> 外圈每 `gradient_accumulation_steps` 步更新一次权重 → 按回调的开关决定何时 log / eval / save。
>
> 读完你应该能回答三个问题：
> 1. 那 4477 行里，**真正每秒都在跑的那 30 行**在哪？
> 2. `gradient_accumulation_steps=4` 到底把什么除以了 4？
> 3. 如果我不想用 `Trainer` 了，哪些零件可以拆下来直接塞进自己的循环？
>
> 源文件：
> - `src/transformers/trainer.py`（**4477 行**，主角）
> - `src/transformers/training_args.py`（2913 行，**112 个 `field()`**，所有超参的唯一入口）
> - `src/transformers/trainer_callback.py`（784 行，16 个回调钩子 + `TrainerState` / `TrainerControl`）
> - `src/transformers/trainer_pt_utils.py`（1342 行，采样器 / `LabelSmoother` / `get_parameter_names`）
> - `src/transformers/trainer_optimizer.py`（615 行，`optim="xxx"` 字符串 → 优化器类的映射表）
> - `src/transformers/trainer_seq2seq.py`（392 行，`Seq2SeqTrainer`）+ `training_args_seq2seq.py`（94 行）
> - `src/transformers/trainer_utils.py`（1256 行，`TrainOutput` / 检查点工具 / `find_executable_batch_size`）

---

## 速读区（TL;DR）

**一句话结论**：`Trainer` 那 4477 行里真正每秒都在跑的是 `_run_epoch` 里的三层 `for` 循环；
「何时 log / eval / save / 停」不在循环里，而由 `DefaultFlowCallback.on_step_end` 通过
`TrainerControl` 的 5 个布尔开关反向控制。

| # | 断言 | 出处（文件:行号） |
|---|------|------------------|
| 1 | **`optimizer.step()` / `lr_scheduler.step()` / `zero_grad()` 都在 `_run_epoch`，不在 `training_step` 里** | `trainer.py:1800` / `:1809` / `:1811` |
| 2 | `TrainingArguments` **实测 112 个 `field()`**（不是常说的 200 个——那是老版本的记忆） | `training_args.py:180-2890`（`awk` 实测） |
| 3 | `num_update_steps_per_epoch` 里的 `int(len_dataloader % gas > 0)` 就是**向上取整**，为最后一个不完整窗口兜底 | `trainer.py:2371-2374` |
| 4 | `training_step` 返回的是 `loss.detach()`（不挂计算图），且 `del inputs` 是刻意写的省显存 | `trainer.py:1978` / `:1948` |
| 5 | 梯度累积有**两套**：老路径 `loss / current_gradient_accumulation_steps`；新路径交给模型的 `num_items_in_batch`（先 `reduction="sum"` 再除） | `trainer.py:1969` / `loss/loss_utils.py:39-45` |
| 6 | 除的是 `current_gradient_accumulation_steps = len(batch_samples)`，**不是 `args.gradient_accumulation_steps`** | `trainer.py:1745` |
| 7 | 分流条件：`model_accepts_loss_kwargs` = 模型显式声明，否则看 `forward` 签名有没有 `VAR_KEYWORD`；Llama 有 `**kwargs` ⇒ 走新路径、**不除 gas** | `trainer.py:503-505` / `models/llama/modeling_llama.py:448` |
| 8 | **`autocast_smart_context_manager()` 直接 `return contextlib.nullcontext()`**——autocast 全权交给 accelerate | `trainer.py:2086` |
| 9 | 混合精度传递链：`ACCELERATE_MIXED_PRECISION` → 被 `fp16` 覆盖 → 再被 `bf16` 覆盖（**两个都开时 `fp16` 赢**） | `training_args.py:1575-1579` |
| 10 | `Trainer` 不用 `accelerator.accumulate()`：它要预取整窗口，故把 `num_steps` 强制写死为 1，累积边界自己管 | `trainer.py:790` / `:1761` |
| 11 | 评估指标统一加前缀（`loss` → `eval_loss`），所以 `metric_for_best_model` 必须写**带前缀**的名字 | `trainer.py:2869-2872` |
| 12 | `warmup_steps` / `logging_steps` / `eval_steps` / `save_steps` 的 **`< 1` 是「比例」语义**，运行时乘总步数 | `trainer_callback.py:156` / `training_args.py:2115` |
| 13 | 全局 batch = `_train_batch_size × gas × (world_size // tp // cp // sp)`——TP/CP/SP 三个维度看到的 batch 是重复的，不能算进去 | `trainer.py:2414` |
| 14 | `save_total_limit=1` 时**永远保留 best、删掉其余全部**；`save_only_model=True` 的 docstring 明说 **Prevents resuming training from checkpoint** | `trainer.py:1888-1892` / `training_args.py:1146` |

**最容易踩的坑 / 反直觉发现**：
- **`training_step` 名字叫 step，但它不更新权重**：函数体内没有 `optimizer.step()` / `zero_grad()`（`sed -n '1907,1979p'` 里 `grep` 无匹配）。正确理解是 `forward_backward_step`。
- **`Trainer` 里那个名字最像答案的 autocast 函数是个空壳**（`:2086` 直接返回 `nullcontext`），源码原话「We rely on accelerate for autocast, hence we do nothing here.」。
- **「`TrainingArguments` 有 200 个字段」已被实测推翻**——当前是 112 个。
- **走新路径时不再除 `gas`**：若覆写了 `compute_loss` 却没把 `num_items_in_batch` 用上，梯度会被放大 `gas` 倍；docstring 明确要求此时把 `self.model_accepts_loss_kwargs` 覆写为 `False`（`trainer.py:2005`）。
- **「什么时候停」不在循环里**，在 `DefaultFlowCallback.on_step_end`（`trainer_callback.py:595`）；`CallbackHandler.__init__` 会检查它，少了它训练会失控（永远不 log、不 save、不停，`:443-449`）。
- **`on_step_begin` 只在累积窗口的第一个 micro-batch 触发**（`trainer.py:1761` 的条件是 `step % gas == 0`），不是每个 micro-batch；每个 micro-batch 都触发的是 `on_substep_end`（`:1826`）。
- 数 `num_items_in_batch` 时 labels 要切 `[..., 1:]`——causal LM 的第 0 位永远不是预测目标，不切会多算 `B` 个。
- **训练循环和「任务类型」解耦**：`Seq2SeqTrainer` 只改了评估路径（`prediction_step` 换 `model.generate`），`train()` 一个字都没改。

**读完应该能回答的问题**：
- 那 4477 行里，真正每秒都在跑的是哪几十行？在哪个函数？
- `gradient_accumulation_steps=4` 到底把什么除以了 4？除的是 `4` 还是别的？
- `fp16=True, bf16=True` 同时打开，最终生效的是哪个？
- 想在 `Trainer` 之外自己写循环，哪些零件可以直接调、哪些必须自己实现？
- 为什么 `Trainer` 不用更省事的 `accelerator.accumulate()`？

---

## 前置知识

- [1A3_from_pretrained.md](./1A3_from_pretrained.md)：`Trainer` **不造模型**，模型是外面传进来的。
  `save_model()` 最终走的就是 1A3 的 `save_pretrained()`。
- [1A5_generate_loop.md](./1A5_generate_loop.md)：`evaluate()` 的 `predict_with_generate` 路径就是调 `generate()`。
  推理循环的对照面。

---

## 一、五个文件的分工

`Trainer` 不是一个文件，是五个文件分工的产物。先建立地图，再往里钻。

| 文件 | 行数 | 负责什么 | 你会为了什么翻开它 |
|------|------|----------|-------------------|
| `trainer.py` | 4477 | **循环本体**：`train` / `_inner_training_loop` / `_run_epoch` / `training_step` / `compute_loss` / `evaluate` / `save_model` | 想改训练行为 |
| `training_args.py` | 2913 | 112 个 dataclass 字段，**所有超参的唯一入口**；`__post_init__` 里做校验和推导 | 想知道某个参数从哪来 |
| `trainer_callback.py` | 784 | `TrainerCallback`（16 个 `on_*` 钩子）、`TrainerState`（要落盘的训练状态）、`TrainerControl`（5 个布尔开关）、`CallbackHandler` | 想插自定义逻辑 |
| `trainer_pt_utils.py` | 1342 | 采样器（`LengthGroupedSampler` / `DistributedSamplerWithLoop`）、`nested_gather` / `nested_concat` 等张量工具、`LabelSmoother`、`get_parameter_names` | 想做数据 / 参数分组 |
| `trainer_optimizer.py` | 615 | `_OPTIMIZER_HANDLERS`（`:596`）：`optim` 字符串 → 构造函数 | 想查某个优化器怎么建的 |
| `trainer_utils.py` | 1256 | `TrainOutput`（`:260`）、`PREFIX_CHECKPOINT_DIR = "checkpoint"`（`:266`）、`get_last_checkpoint`（`:270`）、`sort_checkpoints`（`:282`）、`rotate_checkpoints`（`:342`）、`SchedulerType`（`:561`）、`find_executable_batch_size`（`:938`） | 想找检查点工具 |

**一个结构性事实**：`Trainer` 自己**不实现反向传播、不实现 optimizer、不实现 dataloader**。
后三者分别来自 PyTorch、`torch.optim`、`torch.utils.data`，
而**分布式 + 混合精度的注入全部由 `accelerate.Accelerator` 完成**。
`Trainer` 真正做的事是「调度」：把 `Accelerator`、`DataLoader`、`CallbackHandler`、`TrainerState`
四个对象按正确顺序穿起来。

---

## 二、总流程：从 `train()` 到一次 `optimizer.step()`

这是本篇的核心产出。**这张图从上到下一共四层**，每一层都在源码里有明确的入口行号。

```
┌─ 第 0 层：构造 ───────────────────────────────────────────────────────────────┐
│  Trainer.__init__                                        trainer.py:368       │
│    ├─ 源码里自带 11 步流程注释                            trainer.py:388-399  │
│    ├─ create_accelerator_and_postprocess()  :415  ← Accelerator 在这里诞生    │
│    │     (def :771；_build_accelerator_args def :712，mixed_precision 在 :715)│
│    ├─ CallbackHandler(callbacks, ...)       :578                              │
│    ├─ TrainerControl()  :591 / TrainerState(...)  :592                        │
│    ├─ self._train_batch_size = args.train_batch_size  :609                    │
│    └─ callback_handler.on_init_end(...)     :613                              │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─ 第 1 层：train() ─────────────────┼───────────────────────────────────────────┐
│  Trainer.train()                                         trainer.py:1350      │
│    ├─ if args.gradient_checkpointing:                    :1403               │
│    │     model.gradient_checkpointing_enable()           :1408               │
│    ├─ [resume] _load_from_checkpoint()                   :1446               │
│    ├─ find_executable_batch_size(self._inner_training_loop, ...)      :1452   │
│    └─ inner_training_loop(args=..., resume_from_checkpoint=...)       :1461   │
│         └─ 返回的就是 TrainOutput，train() 直接 return                          │
└───────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─ 第 2 层：_inner_training_loop ────┼───────────────────────────────────────────┐
│  _inner_training_loop()                                  trainer.py:1465      │
│    ├─ get_train_dataloader()                             :1477  (def :881)    │
│    ├─ set_initial_training_values()  :1484-1492  (def :2344)                  │
│    │     → num_train_epochs / num_update_steps_per_epoch / max_steps          │
│    ├─ _init_training_state()         :1494  (def :1558)                       │
│    │     → TrainerState + 断点续训定位 (epochs_trained)                        │
│    ├─ _prepare_for_training()        :1497  (def :1595)                       │
│    │     ├─ create_optimizer()  (def :1171)  → self.optimizer                 │
│    │     ├─ _wrap_model()       (def :2462)  → DDP / DataParallel             │
│    │     ├─ accelerator.prepare(model, optimizer)  ← 分布式 + 精度注入点       │
│    │     └─ create_scheduler()  (def :1247)  → self.lr_scheduler              │
│    ├─ model.zero_grad()                                  :1531                │
│    ├─ callback_handler.on_train_begin(...)               :1533                │
│    ├─ if args.eval_on_start: _evaluate(..., skip_scheduler=True)  :1536       │
│    └─ for epoch in range(epochs_trained, num_train_epochs):  :1538            │
│          ├─ on_epoch_begin(...)                          :1539                │
│          ├─ _run_epoch(...)                              :1540 ────┐          │
│          └─ if control.should_training_stop: break                  │          │
│    _finalize_training()  → TrainOutput                   :1556      │          │
│      (def :1858；内部做 load_best_model_at_end / on_train_end)       │          │
└──────────────────────────────────────────────────────────────────────┼──────────┘
                                                                       │
┌─ 第 3 层：_run_epoch ─────────────────────────────────────────────────┼──────────┐
│  _run_epoch()                                            trainer.py:1693        │
│    epoch_iterator = iter(train_dataloader)               :1727                  │
│    remainder = steps_in_epoch % gas                      :1730                  │
│                                                                                 │
│    ┌── 外层：一次迭代 = 一次 optimizer.step（先预取 num_batches 个 batch）      │
│    ├─ for update_step in range(num_update_steps_trained,     :1737              │
│    │                          num_update_steps_per_epoch):                      │
│    │     num_batches = gas（最后一个 update_step 用 remainder） :1738           │
│    │     batch_samples, num_items_in_batch =                                    │
│    │         self.get_batch_samples(epoch_iterator, num_batches, ...) :1741     │
│    │     current_gradient_accumulation_steps = len(batch_samples)     :1745     │
│    │                                                                            │
│    │     ┌── 内层：一次迭代 = 一个 micro-batch 的 forward + backward             │
│    │     ├─ for i, inputs in enumerate(batch_samples):        :1755             │
│    │     │     step += 1                                      :1756             │
│    │     │     do_sync_step = (step+1) % gas == 0 or 最后一批  :1757            │
│    │     │     accelerator.gradient_state._set_sync_gradients(do_sync_step) :1761
│    │     │     【累积边界】if step % gas == 0:                                    │
│    │     │         on_step_begin(...)                         :1762             │
│    │     │     with no_sync()/nullcontext():  ← 非边界不做梯度同步 :1764-1772     │
│    │     │         tr_loss_step = self.training_step(model, inputs, ...) :1774  │
│    │     │              │  ↳ 见第 4 层                                          │
│    │     │     self._tr_loss += tr_loss_step                   :1788             │
│    │     │     self.current_flos += floating_point_ops(inputs) :1790             │
│    │     │     _track_num_input_tokens(inputs)                :1791             │
│    │     │     │                                                                │
│    │     │     ├─ if do_sync_step:  ← 梯度攒够了，真正更新一次                    │
│    │     │     │     _clip_grad_norm(model)               :1796 (def :2550)      │
│    │     │     │     _get_grad_norm(model, ...)           :1797 (def :2556)      │
│    │     │     │     on_pre_optimizer_step(...)           :1799                │
│    │     │     │     self.optimizer.step()                :1800  ★ 权重变了      │
│    │     │     │     on_optimizer_step(...)               :1801                │
│    │     │     │     learning_rate = _get_learning_rate() :1804                │
│    │     │     │     self.lr_scheduler.step()             :1809  ★ lr 变了       │
│    │     │     │     model.zero_grad()                    :1811  ★ 梯度清零      │
│    │     │     │     self.state.global_step += 1          :1812                │
│    │     │     │     self.state.epoch = epoch + (step+1)/steps_in_epoch :1813  │
│    │     │     │     on_step_end(...)                     :1814                │
│    │     │     │     _maybe_log_save_evaluate(...)        :1815  ↳ 见第八节    │
│    │     │     └─ else: on_substep_end(...)               :1826                │
│    │     │     if should_epoch_stop or should_training_stop: break :1828       │
│    │     └─ ...                                                                │
│    ├─ on_epoch_end(...)                                  :1846                │
│    └─ _maybe_log_save_evaluate(...)                      :1847                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─ 第 4 层：training_step ─────────────────────────────────────────────────────────┐
│  training_step(model, inputs, num_items_in_batch)        trainer.py:1907        │
│    model.train()                                         :1936                 │
│    self.optimizer.train()  (若优化器支持)                 :1937-1938             │
│    inputs = self._prepare_inputs(inputs)                 :1940  → 搬到 device   │
│    with self.compute_loss_context_manager():             :1945                 │
│        loss = self.compute_loss(model, inputs, ...)      :1946                 │
│    del inputs                                            :1948                 │
│    [if 模型不吃 num_items_in_batch] loss /= current_gas    :1969                 │
│    self.accelerator.backward(loss, **kwargs)             :1976  ★ 反向传播       │
│    return loss.detach()                                  :1978                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**这张图里最该记住的三件事**：

1. **`optimizer.step()` 在 `_run_epoch` 里，不在 `training_step` 里**（`:1800`）。
   `training_step` 只负责「算 loss + backward」——名字叫 step，但它**不更新权重**。
   这点和很多手写循环不一样，也是读源码时最容易看错的一处。
2. **三层循环的语义**：`epoch` → `update_step`（= 一次 `optimizer.step()`）→ `micro-batch`（= 一次 forward/backward）。
   三者数量关系：`micro-batch 总数 = steps_in_epoch`，`update_step 总数 = num_update_steps_per_epoch`，且

   ```python
   # trainer.py:2371-2374  set_initial_training_values()
   num_update_steps_per_epoch = max(
       len_dataloader // args.gradient_accumulation_steps
       + int(len_dataloader % args.gradient_accumulation_steps > 0),
       1,
   )
   ```

   注意那个 `int(...)`——**向上取整**。不然最后一个不完整的累积窗口就白跑了。
3. **`_run_epoch` 是较新版本才拆出来的函数**。老版本所有这些内层代码都内联在 `_inner_training_loop` 里，
   现在被拆成「`_inner_training_loop` 管 epoch、`_run_epoch` 管 step」两个方法。

---

## 三、`TrainingArguments`：训练超参的单一入口

`TrainingArguments` 是个 `@dataclass`（`training_args.py:179-180`）。
**实测它有 112 个 `field()` 声明**（不是常说的「200 个」——那是老版本的记忆，
这个版本的字段被合并/删减过）：

```bash
awk 'NR>=180 && NR<=2890 && /^    [a-zA-Z_][a-zA-Z_0-9]*: .*= field\(/ {c++} END {print c}' \
    src/transformers/training_args.py
# → 112
```

字段在源码里按注释分成若干组：Training Duration and Batch Size / Learning Rate & Scheduler /
Optimizer / Mixed Precision / Gradient Checkpointing / Logging & Monitoring / Evaluation /
Saving / Resuming Training / Hub / ……

**别背 112 个。真正的高频参数只有下面这 17 个**：

| 字段 | 行号 | 默认值 | 含义 / 源码要点 |
|------|------|--------|----------------|
| `output_dir` | `:763` | `None` → `__post_init__` 里变成 `"trainer_output"`（`:1493`） | 所有 `checkpoint-*` 往这儿写 |
| `per_device_train_batch_size` | `:769` | `8` | **每张卡**的 batch；全局 batch 见下 |
| `num_train_epochs` | `:770` | `3.0` | 可以是小数（最后一个 epoch 只跑一部分） |
| `max_steps` | `:771` | `-1` | 正数时**覆盖** `num_train_epochs`；流式数据集必须设它 |
| `learning_rate` | `:779` | `5e-5` | 直接进 optimizer 的 `lr`（`trainer.py:1294`） |
| `lr_scheduler_type` | `:780` | `"linear"` | 字符串 → `SchedulerType`（`trainer_utils.py:561`），共 12 种 |
| `warmup_steps` | `:790` | `0` | **`< 1` 时被当成比例**（`get_warmup_steps`，`:2115`） |
| `optim` | `:804` | `"adamw_torch_fused"`（torch≥2.8）否则 `"adamw_torch"` | 字符串键，查 `_OPTIMIZER_HANDLERS` |
| `weight_decay` | `:814` | `0.0` | **自动排除 bias 和 norm 层**（`get_decay_parameter_names`，`trainer.py:1308`） |
| `gradient_accumulation_steps` | `:841` | `1` | 见第六节 |
| `max_grad_norm` | `:858` | `1.0` | `0` 表示不裁剪（判断在 `trainer.py:1795`） |
| `fp16` / `bf16` | `:872` / `:866` | `False` | 见第七节 |
| `gradient_checkpointing` | `:898` | `False` | 在 `train()` 里生效（`trainer.py:1403`），**不是** `__init__` |
| `logging_steps` | `:976` | `500` | **`< 1` 时按总步数比例**（`TrainerState.compute_steps`，`trainer_callback.py:156`） |
| `eval_strategy` | `:1081` | `"no"` | `"no"` / `"steps"` / `"epoch"` |
| `save_steps` | `:1158` | `500` | 同上，`< 1` 当比例 |
| `report_to` | `:1034` | `"none"` | 决定 `get_reporting_integration_callbacks()` 附加哪些回调（`trainer.py:568`） |

**一个容易被忽略的「比例语义」**：`warmup_steps` / `logging_steps` / `eval_steps` / `save_steps`
都接受 `(0, 1)` 的小数，在运行时被**乘上总步数**转成绝对值：

```python
# trainer_callback.py:156  TrainerState.compute_steps()
for step_kind in ("logging", "eval", "save"):
    num_steps = getattr(args, f"{step_kind}_steps")
    if num_steps is not None:
        if num_steps < 1:
            num_steps = math.ceil(max_steps * num_steps)
        setattr(self, f"{step_kind}_steps", num_steps)
```

`warmup_steps` 走的是同构逻辑，但实现在 `TrainingArguments` 自己身上：

```python
# training_args.py:2115  get_warmup_steps()
warmup_steps = (
    int(self.warmup_steps) if self.warmup_steps >= 1 else math.ceil(num_training_steps * self.warmup_steps)
)
```

**全局 batch size 的完整公式**（别只记 `per_device × n_gpu × gas`）：

```python
# trainer.py:2414  get_total_train_batch_size()
dp_world_size = args.world_size // self.get_tp_size() // self.get_cp_size() // self.get_sp_size()
return self._train_batch_size * args.gradient_accumulation_steps * dp_world_size
```

也就是说：`world_size` 要先除以 TP（张量并行）、CP（上下文并行）、SP（序列并行）三个维度
才是真正的 data-parallel 份数。这三个维度看到的 batch 是**重复的**，不能算进去。

`__post_init__`（`:1490`）里做了大量推导和校验，值得知道的几条：

| 位置 | 干什么 |
|------|--------|
| `:1575-1579` | `mixed_precision` 先读环境变量 `ACCELERATE_MIXED_PRECISION`，再被 `fp16` / `bf16` 覆盖 |
| `:1622` | 访问 `self.device` —— 触发 `_setup_devices`（def 在 `:1804`），分布式环境在这里初始化 |
| `:1660-1681` | DeepSpeed 配置解析，`deepspeed_plugin` 在这里建 |
| `_validate_args()`（`:1690` 起） | `logging_steps > 1` 必须是整数；`eval_strategy="steps"` 时 `eval_steps` 必须给 |

---

## 四、训练循环的一步（step）

### 4.1 `training_step` 的真实代码（`trainer.py:1907`）

这是**每秒都在跑的那 30 行**。去掉 docstring 和分布式分支后：

```python
# trainer.py:1935-1978（省略未标注的行）
with cp_context():                                    # 上下文并行的空壳
    model.train()                                     # :1936
    ...
    inputs = self._prepare_inputs(inputs)             # :1940 搬到 device
    ...
    with self.compute_loss_context_manager():         # :1945 混合精度上下文
        loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)  # :1946

    del inputs                                        # :1948 尽早释放
    ...
    if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
        loss = loss / self.current_gradient_accumulation_steps    # :1969 ★

    if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
        kwargs["scale_wrt_gas"] = False                # :1974 DeepSpeed 自己会缩放

    self.accelerator.backward(loss, **kwargs)          # :1976 ★ 反向传播

    return loss.detach()                               # :1978
```

**注意四处**：

- **没有 `optimizer.step()`、没有 `optimizer.zero_grad()`**。它们都在 `_run_epoch`（`:1800` / `:1811`）。
  所以 `training_step` 的正确理解是 **`forward_backward_step`**。
- **`self.accelerator.backward(loss)` 而不是 `loss.backward()`**。这一步里 accelerate 干了三件事：
  ① fp16 时先把 loss 乘上 scaler；② 反向；③ 在 GA 边界上 unscale 梯度。
- **`del inputs`（`:1948`）是刻意写的**。配合可选的 `torch_empty_cache_steps`（判断在 `:1949-1955`）
  手动 `clear_device_cache()`，为了在长序列训练里省显存。
- **`loss.detach()` 而不是 `loss`**：返回的是「日志用」的标量副本，**不挂在计算图上**。
  不然 `_tr_loss += tr_loss_step` 会让每步的图都活着，显存直接爆掉。

### 4.2 一个 step 的完整时序

以 `gradient_accumulation_steps=2`、`steps_in_epoch=6` 为例，`step` 从 0 开始计数：

| step | `do_sync_step` | `no_sync()` | 做的事 |
|------|---------------|-------------|--------|
| 0 | False | **是**（梯度留本地） | `training_step` → `_tr_loss +=` → `on_substep_end` |
| 1 | True | 否 | `training_step` → `clip_grad_norm` → `optimizer.step()` → `lr_scheduler.step()` → `zero_grad()` → `global_step=1` → `on_step_end` → `_maybe_log_save_evaluate` |
| 2 | False | 是 | 同 step 0 |
| 3 | True | 否 | 更新，`global_step=2` |
| 4 | False | 是 | 同 step 0 |
| 5 | True | 否 | 更新，`global_step=3`（`(step+1)==steps_in_epoch` 强制同步） |

`do_sync_step` 的定义（`:1757`）：

```python
do_sync_step = (step + 1) % self.args.gradient_accumulation_steps == 0 or (step + 1) == steps_in_epoch
```

**`or (step+1) == steps_in_epoch` 是必须的**：如果 `steps_in_epoch` 不能被 `gas` 整除
（比如 7 个 batch、`gas=2`），最后一个不完整的累积窗口也必须同步一次，
否则那批梯度永远留在本地、永远不会进 `optimizer.step()`。

`self.accelerator.gradient_state._set_sync_gradients(do_sync_step)`（`:1761`）是**手动**通知 accelerate 的，
旁边注释写得很明确——「Since we perform prefetching, we need to manually set sync_gradients」。
`Trainer` 不用 accelerate 的 GA 插件：`create_accelerator_and_postprocess` 里把
`grad_acc_kwargs["num_steps"] = 1` 强制写死（`trainer.py:790`），原因就是累积边界由自己控制。

### 4.3 为什么不用 `accelerator.accumulate()`

accelerate 提供了 `with accelerator.accumulate(model):` 这种更省事的写法。
`Trainer` 不用它，是因为它要**预取**：`get_batch_samples(epoch_iterator, num_batches, ...)`（`:1741`）
一次就把 `num_batches` 个 batch 抓出来放进列表，好处是

① 能提前算 `num_items_in_batch`（需要看整窗口的 labels）；
② 能算出本窗口真实的 `current_gradient_accumulation_steps`（最后一个窗口可能不足 `gas`）。

代价就是同步开关得自己管，并且 `no_sync` 的上下文要在循环里显式包起来（`:1764-1772`）。

---

## 五、`compute_loss`：loss 从哪来

### 5.1 默认路径：模型自己算，`Trainer` 只是取出来

```python
# trainer.py:2011-2020（简化）
if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
    labels = inputs.pop("labels")                      # :2012
else:
    labels = None
if self.model_accepts_loss_kwargs:
    kwargs = {}
    if num_items_in_batch is not None:
        kwargs["num_items_in_batch"] = num_items_in_batch
    inputs = {**inputs, **kwargs}                      # :2019
outputs = model(**inputs)                              # :2020 ★ 模型 forward
```

**关键点：`labels` 是原样传进 `model(**inputs)` 的**（只在有 `label_smoother` 或
`compute_loss_func` 时才被 `pop` 出来）。所以「shift + cross_entropy」发生在**模型内部**。

Llama 那边（`models/llama/modeling_llama.py:483-484`）：

```python
loss = None
if labels is not None:
    loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
```

`self.loss_function` 是 `PreTrainedModel` 的属性（`modeling_utils.py:4708-4720`），
按 `self.loss_type` 去查 `LOSS_MAPPING`（`loss/loss_utils.py:172`），Llama 落到 `ForCausalLMLoss`：

```python
# loss/loss_utils.py:49-71
def ForCausalLMLoss(logits, labels, vocab_size, num_items_in_batch=None,
                    ignore_index=-100, shift_labels=None, **kwargs):
    logits = logits.float()                       # :59 上采到 fp32，避免半精度下 CE 溢出
    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)   # :63
        shift_labels = labels[..., 1:].contiguous()                      # :64
    # Flatten the tokens
    logits = logits.view(-1, vocab_size)                                 # :67
    shift_labels = shift_labels.view(-1)                                 # :68
    shift_labels = shift_labels.to(logits.device)
    loss = fixed_cross_entropy(logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)  # :70
    return loss
```

**这是「decoder-only 的 loss 长什么样」的标准答案**：
`logits[i]` 预测 `labels[i+1]`，即先 `pad` 再往后错一位，然后把 `[B, S, V]` 和 `[B, S]`
都摊平成 `[B*S, V]` 和 `[B*S]`，直接 `cross_entropy`。`ignore_index=-100` 的位置（prompt、padding）不计 loss。

按 `Trainer` 自己的口径（`trainer.py:525-527`），它在 `__init__` 里就先把「谁会 shift」这件事算出来了：

```python
# trainer.py:525-527
self._loss_shifts_labels = LOSS_MAPPING.get(
    getattr(model_to_inspect, "loss_type", None)
) is ForCausalLMLoss and not getattr(getattr(model_to_inspect, "config", None), "is_encoder_decoder", False)
```

——**只有 causal LM 才 shift labels**。Encoder-decoder（T5 那种）的 loss 是对齐的、不 shift，
因为它的 `decoder_input_ids` 已经在外面右移过了。这个布尔量后面会用来数 token（见第六节）。

### 5.2 四条自定义 loss 的入口

| 入口 | 位置 | 粒度 | 什么时候用 |
|------|------|------|-----------|
| `Trainer(compute_loss_func=...)` | 构造参数，用在 `trainer.py:2023-2033` | **整个训练** | 换 loss 函数本身（自定义 KL、对比学习等） |
| 继承 `Trainer` 覆写 `compute_loss` | `trainer.py:1980` | 整个训练 | 要改「怎么调模型」而不只是 loss |
| `model.loss_function = my_fn` | setter 在 `modeling_utils.py:4723` | **单模型** | 换掉 `LOSS_MAPPING` 里那一个函数 |
| `config.loss_type = "..."` | `modeling_utils.py:4713` | 单模型 | 从 `LOSS_MAPPING` 里挑另一个 |

`compute_loss_func` 的调用点在 `trainer.py:2029`：

```python
if self.compute_loss_func is not None:
    ...
    loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
```

**它拿到的 `outputs` 已经是模型输出**，`labels` 是刚才 `pop` 出来的
（如果没设 `label_smoother`，这里就是 `None`——源码为此专门打了个 warning，`:2025-2028`）。

还有一条「半自定义」的入口是 `label_smoothing_factor`（`training_args.py:861`）：
它会建一个 `LabelSmoother`（`trainer_pt_utils.py:436`），让 `Trainer` 接管 loss 计算
（这就是 `labels` 被 `pop` 的原因），并且**对 causal LM 自动加 `shift_labels=True`**（`trainer.py:2043`）：

```python
if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
    loss = self.label_smoother(outputs, labels, shift_labels=True, num_items_in_batch=num_items_in_batch)
else:
    loss = self.label_smoother(outputs, labels, num_items_in_batch=num_items_in_batch)
```

最后一条兜底路径在 `:2048-2053`：如果模型没返回 loss，`Trainer` 会**抛错并列出 outputs 的 key**
（而不是悄悄用 logits）：

```python
if isinstance(outputs, dict) and "loss" not in outputs:
    raise ValueError(
        "The model did not return a loss from the inputs, only the following keys: "
        f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
    )
loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
```

---

## 六、梯度累积：到底把什么除以了什么

这是全篇最容易记错的一节。**历史上 `Trainer` 只有一种做法，现在有两种，靠一个条件分流。**

### 6.1 老做法：loss 除以 `gradient_accumulation_steps`

```python
# trainer.py:1967-1969
if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
    loss = loss / self.current_gradient_accumulation_steps
```

**为什么要除**：PyTorch 的梯度是**累加**的。一个窗口里跑 `gas` 次 backward，
参数 `w` 的梯度会变成 `∂L₁/∂w + ∂L₂/∂w + … + ∂L_gas/∂w`，
而「大 batch 的梯度」应该是**平均** `(1/gas)·Σ∂Lᵢ/∂w`。
在每次 backward 前把 loss 缩小，累加的结果就等价于平均。

**为什么除的是 `current_gradient_accumulation_steps` 而不是 `args.gradient_accumulation_steps`**：
最后一个不完整窗口可能只有 3 个 batch 而不是 4 个。除 4 会让最后一步的梯度被人为缩小 25%。
`current_gradient_accumulation_steps` 在 `_run_epoch:1745` 被设成本窗口真实的 batch 数：

```python
self.current_gradient_accumulation_steps = len(batch_samples)
```

### 6.2 新做法：`num_items_in_batch`（按有效 token 数加权）

`loss / gas` 有个隐藏假设：**每个 batch 的有效 token 数大致相等**。
但 NLP 里不是——一批全是短句（labels 大部分是 `-100`）和一批全是长句，
权重却相同，等价于给短样本过高权重。

现代做法是**用整窗口的有效 token 总数当分母**：

```python
# loss/loss_utils.py:32-46
def fixed_cross_entropy(source, target, num_items_in_batch=None, ignore_index=-100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"   # :39 ★ 先求和，不平均
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch                                # :45 ★ 再除以有效 token 数
    return loss
```

`reduction="sum"` + 除以 `num_items_in_batch` = **按 token 平均**，跨 batch 都是同一个分母，
所以和 batch 边界无关。`num_items_in_batch` 由 `Trainer._get_num_items_in_batch`（`:2156`）算出来：

```python
# trainer.py:2186-2197（核心几行）
labels_for_count = [
    batch["shift_labels"] if "shift_labels" in batch
    else batch["labels"][..., 1:] if self._loss_shifts_labels
    else batch["labels"]
    for batch in batch_samples
]
num_items_in_batch = sum(labels.ne(-100).sum() for labels in labels_for_count)
```

即「**整个累积窗口里非 `-100` 的 label 个数**」。注意 `[..., 1:]`——因为 causal LM 的第 0 位
永远不是预测目标（见第五节），不切掉就会多算 `B` 个。

**分流的条件就是那一个 `if`**（`trainer.py:1967`）：

```python
if (not self.model_accepts_loss_kwargs or num_items_in_batch is None) and self.compute_loss_func is None:
    loss = loss / self.current_gradient_accumulation_steps
```

`self.model_accepts_loss_kwargs` 在 `__init__` 里探测（`trainer.py:499-506`）：
模型显式声明了就用它的值，否则**检查 `forward` 签名里有没有 `VAR_KEYWORD`**：

```python
# trainer.py:503-505
self.model_accepts_loss_kwargs = any(
    k.kind == inspect.Parameter.VAR_KEYWORD for k in forward_params.values()
)
```

Llama 的 `forward` 结尾是 `**kwargs: Unpack[TransformersKwargs]`（`modeling_llama.py:448`），所以是 `True`
——**绝大多数现代模型都走新路径，不除 `gas`**。

| | 老路径 | 新路径 |
|---|---|---|
| 触发条件 | 模型不吃 `num_items_in_batch` | 模型接受 `num_items_in_batch` |
| loss 分母 | `current_gradient_accumulation_steps` | 窗口内有效 token 数 |
| 代码位置 | `trainer.py:1969` | `loss/loss_utils.py:45` |
| 谁负责 | `Trainer` | 模型内部的 `ForCausalLMLoss` |
| 精度 | batch 长度不均时有偏 | 无偏 |

**一个坑**：如果你覆写了 `compute_loss` 但没把 `num_items_in_batch` 用上，
`Trainer` 会误以为模型已经处理好了，于是**也不除 `gas`**，结果梯度被放大 `gas` 倍。
docstring 里明确写了（`trainer.py:2005`）：

> If you are not using `num_items_in_batch` when computing your loss, make sure to overwrite
> `self.model_accepts_loss_kwargs` to `False`.

---

## 七、混合精度：`autocast` 在哪生效

这一节有个**反直觉的发现**：`Trainer` 里那个名字最像答案的函数，其实是个空壳。

```python
# trainer.py:2069-2086
def compute_loss_context_manager(self) -> contextlib.ExitStack:
    """A helper wrapper to group together context managers."""
    ctx_stack = contextlib.ExitStack()                           # :2073
    autocast_ctx = self.autocast_smart_context_manager()         # :2075
    if not isinstance(autocast_ctx, contextlib.nullcontext):     # :2076 ★ 恒为 False
        ctx_stack.enter_context(autocast_ctx)
    return ctx_stack                                             # :2079

def autocast_smart_context_manager(self, cache_enabled: bool | None = True) -> contextlib.AbstractContextManager:
    """
    A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
    arguments, depending on the situation. We rely on accelerate for autocast, hence we do nothing here.
    """
    return contextlib.nullcontext()                              # :2086 ★ 直接返回空上下文
```

**「We rely on accelerate for autocast, hence we do nothing here.」**（`:2084`）
——这是这个版本最重要的一句话。历史上 `Trainer` 自己调 `torch.cuda.amp.autocast`，
现在**整个 autocast 职责移交给了 accelerate**。

那 `fp16` / `bf16` 到底从哪儿进去的？三步：

```
① training_args.py:1575-1579
     self.mixed_precision = os.environ.get("ACCELERATE_MIXED_PRECISION", "no")
     if self.fp16:   self.mixed_precision = "fp16"
     elif self.bf16: self.mixed_precision = "bf16"
        │
        ▼
② trainer.py:715   _build_accelerator_args()
     args = { "mixed_precision": self.args.mixed_precision, "deepspeed_plugin": ... }
        │  （由 create_accelerator_and_postprocess 在 trainer.py:415 调用，def 在 :771）
        ▼
③ Accelerator(mixed_precision="bf16", ...)
     accelerate 内部在 prepare_model() 时把 model.forward 包进 autocast
     —— 训练时每次 model(**inputs) 自动在半精度里算，输出再转回 fp32
```

**所以 `Trainer` 侧留下的只有 `ExitStack` 这个壳**（`:2069-2079`）。它的真实作用是：
① 给子类一个**覆写点**（老代码继承 `Trainer` 覆写 `autocast_smart_context_manager` 仍然有效）；
② 把「上下文管理器集合」这个概念留住——将来要加别的上下文（比如 CP）往 `ExitStack` 里塞就行。

`TrainingArguments` 里跟精度有关的参数要分清：

| 参数 | 行号 | 含义 |
|------|------|------|
| `fp16` / `bf16` | `:872` / `:866` | **混合精度训练**：权重保持 fp32，前向/反向用半精度。FP16 还需要 GradScaler，BF16 不需要 |
| `bf16_full_eval` / `fp16_full_eval` | `:878` / `:884` | **评估也用半精度**（连权重都转半精度，更省显存）。`__init__` 会因此推迟设备搬运（`:471`），`train()` 里再补一次（`trainer.py:1399`） |
| `tf32` | `:890` | Ampere+ 上的 19-bit 格式，只影响 matmul 累加精度，**不改 dtype**。在 `__post_init__` 里处理（`:1624-1647`） |

**`fp16` 与 `bf16` 的选择**：`__post_init__` 里是 `if fp16: ... elif bf16:` 的顺序（`:1577-1579`），
两个都开时 `fp16` 赢。源码注释也建议优先 `bf16`（`:867`：「Generally preferred over FP16 due to better numerical stability」）。

---

## 八、`TrainerCallback` 钩子

### 8.1 四个对象的分工

| 类 | 行号 | 角色 |
|----|------|------|
| `TrainerCallback` | `trainer_callback.py:295` | **钩子基类**，16 个 `on_*` 方法全是空实现（`:346-427`） |
| `TrainerState` | `:35` | **要落盘的状态**：`global_step` / `epoch` / `log_history` / `best_metric` / `num_input_tokens_seen`。存进 `trainer_state.json`（`save_to_json`，`:143`） |
| `TrainerControl` | `:234` | **回调反控循环的 5 个开关**：`should_training_stop` / `should_epoch_stop` / `should_save` / `should_evaluate` / `should_log`（`:262-266`） |
| `CallbackHandler` | `:429` | 持有回调列表，负责逐个调用并汇总 `control`（`call_event`，`:543`） |
| `ExportableState` | `:189` | 标记「状态可序列化」的回调基类，`EarlyStoppingCallback` 是唯一内置用户 |

**`TrainerControl` 是这套设计的精髓**：回调**不能**直接 `break` 训练循环，
它只能把 `control.should_training_stop = True`。循环每轮检查这个标志。

**复位规则写在 `CallbackHandler` 的覆写里**（这点很容易看漏）：

```python
# trainer_callback.py:504-508
def on_step_begin(self, args, state, control, **kwargs):
    control.should_log = False        # ★ 每个 step 开头先把三个开关清零
    control.should_evaluate = False
    control.should_save = False
    return self.call_event("on_step_begin", args, state, control, **kwargs)

# :522-533（节选）
def on_evaluate(self, ...): control.should_evaluate = False; ...
def on_save(self, ...):     control.should_save = False;     ...
def on_log(self, ...):      control.should_log = False;      ...
```

——**开关是「一次性」的**：消费它的回调（`on_evaluate` / `on_save` / `on_log`）负责关掉它。
而 `should_training_stop` **永不复位**（docstring `:243`：「If `True`, this variable will not be set back to `False`.
The training will just stop.」），`should_epoch_stop` 只在 `on_epoch_begin` 复位（`:497-499`）。

### 8.2 钩子时机表

⚠️ **执行顺序里有个陷阱**：`on_step_begin` 的条件是
`if step % self.args.gradient_accumulation_steps == 0`（`trainer.py:1761`），
所以它**只在累积窗口的第一个 micro-batch 触发**，不是每个 micro-batch 都触发。
每个 micro-batch 都触发的是 `on_substep_end`（在非同步分支里，`:1826`）。

| 钩子 | 定义行 | 触发点（`trainer.py`） | epoch | optimizer.step | micro-batch |
|------|--------|----------------------|:-----:|:--------------:|:-----------:|
| `on_init_end` | `:346` | `__init__` 末尾 `:613` | — | — | — |
| `on_train_begin` | `:351` | `_inner_training_loop:1533` | — | — | — |
| `on_epoch_begin` | `:361` | `:1539` | ✅ | | |
| `on_step_begin` | `:371` | `:1762` | | ✅（窗口首个） | |
| `on_pre_optimizer_step` | `:377` | `:1799` | | ✅ | |
| `on_optimizer_step` | `:382` | `:1801` | | ✅ | |
| `on_step_end` | `:392` | `:1814` | | ✅ | |
| `on_substep_end` | `:387` | `:1826` | | | ✅（非边界） |
| `on_log` | `:413` | `Trainer.log():3932` | | 视 `should_log` | |
| `on_evaluate` | `:398` | `evaluate():2662` | | 视 `should_evaluate` | |
| `on_save` | `:408` | `_maybe_log_save_evaluate:2136` | | 视 `should_save` | |
| `on_prediction_step` | `:418` | `evaluation_loop:2793` | 每次 eval 的每个 batch | | |
| `on_predict` | `:403` | `predict()` | — | — | — |
| `on_epoch_end` | `:366` | `:1846` | ✅ | | |
| `on_train_end` | `:356` | `_finalize_training:1894` | — | — | — |
| `on_push_begin` | `:423` | `push_to_hub` 路径 | — | — | — |

**`on_step_end` 是最重要的一个**——因为**「要不要 log / eval / save / 停」是在这里决定的**，
而不是循环自己判断（见下一节）。

`CallbackHandler.call_event`（`:543`）的规则：**按注册顺序**依次调用，
每个回调返回的 `control`（返回 `None` 就用旧的）成为下一个回调的输入。

### 8.3 `DefaultFlowCallback`：训练节奏的调度器（`trainer_callback.py:563`）

它只有两个方法，却**决定了整个训练的节奏**：

```python
# trainer_callback.py:568-605（节选）
def on_step_end(self, args, state, control, **kwargs):
    # Log
    if state.global_step == 1 and args.logging_first_step:
        control.should_log = True
    if args.logging_strategy == IntervalStrategy.STEPS and state.global_step % state.logging_steps == 0:
        control.should_log = True
    # Evaluate
    if (args.eval_strategy == IntervalStrategy.STEPS
            and state.global_step % state.eval_steps == 0
            and args.eval_delay <= state.global_step):
        control.should_evaluate = True
    # Save
    if (args.save_strategy == SaveStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0):
        control.should_save = True
    # End training
    if state.global_step >= state.max_steps:          # :595
        control.should_training_stop = True
        # ... 并补一次 eval / save，避免最后一步的指标丢失
    return control
```

**「训练什么时候停」不在循环里，在这里**（`:595`）。循环只负责检查 `control.should_training_stop`。

⚠️ `CallbackHandler.__init__` 会检查有没有 `DefaultFlowCallback`，没有就 warn
（`trainer_callback.py:443-449`）——**少了它训练会失控**（永远不 log、不 save、不停）。

### 8.4 其他内置回调

| 类 | 行号 | 干什么 |
|----|------|--------|
| `DefaultFlowCallback` | `:563` | 上面那个。`DEFAULT_CALLBACKS` 的唯一成员（`trainer.py:184`） |
| `ProgressCallback` | `:624` | tqdm 进度条。`DEFAULT_PROGRESS_CALLBACK`（`trainer.py:185`）；notebook 里换成 `NotebookProgressCallback`（`trainer.py:190`） |
| `PrinterCallback` | `:698` | 裸打印版。`args.disable_tqdm=True` 时替掉进度条（`trainer.py:581`） |
| `EarlyStoppingCallback` | `:711` | 唯一内置的**状态化**回调（继承 `ExportableState`），状态能存进 `trainer_state.json` 并读回来 |

装配顺序在 `__init__` 里（`trainer.py:568-581`）：

```python
default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)   # :568
...
self.callback_handler = CallbackHandler(callbacks, self.model, self.processing_class, ...)         # :578
self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)        # :581
```

`report_to` 决定附加哪些**外部**回调：`get_reporting_integration_callbacks(self.args.report_to)`
去 `integrations/` 里找 wandb / tensorboard / mlflow 等对应的回调类。

---

## 九、评估与保存

### 9.1 `_maybe_log_save_evaluate`：三个开关的消费点（`trainer.py:2088`）

```python
# trainer.py:2100-2136（简化）
if self.control.should_log and self.state.global_step > self._globalstep_last_logged:   # :2100
    logs["loss"] = tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged)
    logs["learning_rate"] = self._get_learning_rate()
    self.log(logs, start_time)                                     # :2124

metrics = None
if self.control.should_evaluate:                                   # :2127
    metrics = self._evaluate(trial, ignore_keys_for_eval)          # :2128
    is_new_best_metric = self._determine_best_metric(...)          # :2129
    if self.args.save_strategy == SaveStrategy.BEST:               # :2131
        self.control.should_save = is_new_best_metric

if self.control.should_save:                                       # :2134
    self._save_checkpoint(model, trial)                            # :2135
    self.control = self.callback_handler.on_save(self.args, self.state, self.control)  # :2136
```

**注意 `SaveStrategy.BEST` 的特殊性**（`:2131-2132`）：
它不是时间驱动的，而是**指标驱动的**——`save_strategy="best"` 时
`control.should_save` 被 `_determine_best_metric()` 的布尔结果赋值。

### 9.2 `evaluate()` 流程

```
_maybe_log_save_evaluate()                          trainer.py:2088
  └─ if self.control.should_evaluate:                        :2127
       metrics = self._evaluate(trial, ignore_keys_for_eval) :2128  (def :3045)
         ├─ self.evaluate(...)                       (def :2569)
         │    ├─ eval_dataloader = self.get_eval_dataloader(eval_dataset)   :2629
         │    ├─ output = self.evaluation_loop(...)  (def :2668)  :2635
         │    │    ├─ model = self._wrap_model(self.model, training=False)  :2689
         │    │    ├─ for step, inputs in enumerate(dataloader):
         │    │    │     loss, logits, labels = self.prediction_step(...)   (def :2936)
         │    │    │        └─ with torch.no_grad():                        :2990
         │    │    │             loss, outputs = self.compute_loss(..., return_outputs=True)  :3018
         │    │    │     on_prediction_step(...)                           :2793
         │    │    └─ metrics = self.compute_metrics(EvalPrediction(...))  :2852
         │    ├─ self.log(output.metrics)                                 :2657
         │    └─ on_evaluate(...)                                          :2662
         ├─ _report_to_hp_search(...)                :3053
         └─ ReduceLROnPlateau / GreedyLR: lr_scheduler.step(metric)   :3055-3064
```

**两个关键点**：

1. **评估不建计算图，但用的是同一套 loss 逻辑**：`prediction_step` 里是 `with torch.no_grad():`（`:2990`），
   里面仍然调 `self.compute_loss(..., return_outputs=True)`（`:3018`）。
2. **所有指标会被统一加前缀**（`evaluation_loop:2869-2872`）：

   ```python
   for key in list(metrics.keys()):
       if not key.startswith(f"{metric_key_prefix}_"):
           metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
   ```

   `loss` → `eval_loss`，你 `compute_metrics` 返回的 `accuracy` → `eval_accuracy`。
   所以 `metric_for_best_model` 必须写**带前缀**的名字（`"eval_loss"` / `"eval_accuracy"`）。

需要 metric 才能 step 的两个 scheduler 是延后处理的，都在 `_evaluate`（def `:3045`）里：

```python
# trainer.py:3055-3064
if isinstance(self.lr_scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, GreedyLR)) and not skip_scheduler:
    metric_to_check = self.args.metric_for_best_model
    if not metric_to_check.startswith("eval_"):
        metric_to_check = f"eval_{metric_to_check}"
    self.lr_scheduler.step(metrics[metric_to_check])
```

——这就是 `_run_epoch:1806-1809` 里那个「如果不是 Plateauf 类才立刻 step」判断的原因：

```python
if not self.accelerator.optimizer_step_was_skipped:
    # Delay optimizer scheduling until metrics are generated
    if not isinstance(self.lr_scheduler, (torch.optim.lr_scheduler.ReduceLROnPlateau, GreedyLR)):
        self.lr_scheduler.step()
```

### 9.3 `save_model` 与 `checkpoint-*` 目录

`save_model()`（`trainer.py:3811`）按后端分五条路，**最终都落到 `_save()`**（`:3869`）：

| 后端 | `save_model` 行为 | 行号 |
|------|------------------|------|
| 普通 | `self._save(output_dir)` | `:3862` |
| DeepSpeed | `accelerator.get_state_dict()` 攒出完整 state_dict 再 `_save`；攒不出来就降级成 `save_checkpoint` + 删假权重 | `:3846` / `:3859` |
| FSDP | `"FULL_STATE_DICT" in state_dict_type` 时才 `_save` | `:3833` |
| SageMaker MP | 额外 `Path(...user_content.pt).touch()` | `:3831` |
| TPU | `save_tpu_checkpoint(...)` | `:3822` |

`_save()` 做三件事（`:3869-3907`）：

```python
self.model.save_pretrained(output_dir, state_dict=state_dict)   # :3893 ← 走 1A3 那条路
...
self.processing_class.save_pretrained(output_dir)               # :3896 tokenizer 也一起存
...
torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))  # :3906
```

**分片 + `safetensors` 都由 `save_pretrained` 决定**（`model.safetensors.index.json` + 多个分片文件）；
如果 `self.model` 不是 `PreTrainedModel`，才会走手写的 `safetensors.torch.save_file`（`:3889`）。

**`checkpoint-*` 目录结构**（由 `_save_checkpoint`，`trainer.py:3096` 组织）：

```
output_dir/
├── checkpoint-500/        ← f"{PREFIX_CHECKPOINT_DIR}-{global_step}"
│   │                        （:3103；PREFIX_CHECKPOINT_DIR = "checkpoint"，trainer_utils.py:266）
│   ├── model.safetensors    ← ① save_model() → _save()（可能分成多个分片 + index.json）
│   ├── config.json
│   ├── tokenizer.json / tokenizer_config.json ...
│   ├── training_args.bin    ← ③ TRAINING_ARGS_NAME（trainer.py:243）
│   ├── optimizer.pt         ← ② OPTIMIZER_NAME（trainer.py:245）
│   ├── scheduler.pt         ← ② SCHEDULER_NAME（trainer.py:248）
│   ├── scaler.pt            ← ② SCALER_NAME（trainer.py:246），只有 fp16 才有
│   ├── rng_state.pth        ← ② _save_rng_state（def :3203）
│   └── trainer_state.json   ← ② TRAINER_STATE_NAME（trainer.py:244）
├── checkpoint-1000/
└── ...
```

`_save_checkpoint` 里 ② 这一段的条件（`:3131-3136`）：

```python
if not self.args.save_only_model:
    self._save_optimizer_and_scheduler(output_dir)      # :3133 → optimizer.pt / scheduler.pt
    self._save_scaler(output_dir)                       # :3134 → scaler.pt
    self._save_rng_state(output_dir)                    # :3136 → rng_state.pth
```

`trainer_state.json` 单独存（`:3150`），且会把 `ExportableState` 回调的状态一起打包（`:3141-3149`）。

**四个「训练状态文件」各自的作用**（常量定义在 `trainer.py:243-248`）：

| 文件 | 存什么 | 少了它会怎样 |
|------|--------|-------------|
| `optimizer.pt` | `optimizer.state_dict()`（`:3305`） | Adam 的一阶/二阶动量丢失 → 续训后 loss 抖动 |
| `scheduler.pt` | `lr_scheduler.state_dict()`（`:3317`） | lr 从 0 重新 warmup → 相当于重新开始 |
| `trainer_state.json` | `TrainerState`：`global_step` / `epoch` / `log_history` / `best_metric` / `best_model_checkpoint` | **续训时找不到位置**。`_init_training_state`（`:1558`）第一件事就是读它（`:1575-1577`） |
| `rng_state.pth` | 各 device 的随机数状态 | dataloader 的 shuffle 顺序和 dropout 无法复现 |

`save_only_model=True`（`training_args.py:1146`）时只写模型权重——
docstring 明说：「Save only model weights, not optimizer/scheduler/RNG state.
**Prevents resuming training from checkpoint.**」

**`save_total_limit` 的轮转**：`_save_checkpoint` 末尾（`:3158-3163`）调 `rotate_checkpoints`
（`trainer_utils.py:342`）；`_finalize_training` 里还会为一个特殊情况再兜一次底（`:1888-1892`）：

```python
# trainer.py:1888-1892
if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
    for checkpoint in checkpoints_sorted:
        if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
            shutil.rmtree(checkpoint, ignore_errors=True)
```

即 `save_total_limit=1` 时**永远保留 best，删掉其余全部**。

**最佳模型的回填**在 `_finalize_training` 里（`:1880-1881`）：

```python
if self.args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
    self._load_best_model()          # def :3472
```

注意是**训练全部结束后才把 best 权重加载回内存**，不是每个 step 都回滚。

---

## 十、`trainer_seq2seq.py` 的存在意义

一句话：**`Seq2SeqTrainer` 只做了一件事——把评估时的「算 loss」换成「`model.generate()`」**，
因为 encoder-decoder（翻译、摘要）的评估指标（BLEU / ROUGE）必须建立在**自回归解码的输出**上，
而不是 teacher-forcing 的 logits 上。

具体差异只有四处：

| 差异 | 位置 |
|------|------|
| `Seq2SeqTrainingArguments` 多出 `sortish_sampler` / `predict_with_generate` / `generation_max_length` / `generation_num_beams` / `generation_config` 五个字段 | `training_args_seq2seq.py:55-77` |
| 构造时用 `args.generation_config` 覆盖 `model.generation_config`（还有个 `load_generation_config` 静态方法做校验） | `trainer_seq2seq.py:91-93` |
| `prediction_step` 覆写：`predict_with_generate=True` 时改调 `self.model.generate(**generation_inputs, **gen_kwargs)` | `trainer_seq2seq.py:329` |
| `evaluate()` 覆写：把 `generation_max_length` / `generation_num_beams` 塞进 `gen_kwargs`，并把 `gather_function` 换成**不去重**的 `accelerator.gather`（生成结果不能按样本去重） | `trainer_seq2seq.py:159-194`（`predict` 里同样处理，`:254`） |

`train()` **一个字都没改**——训练循环是共用的。这印证了一件事：
`Trainer` 的训练循环和「任务类型」是解耦的。

（顺带：`generate()` 的内部机制见 [1A5_generate_loop.md](./1A5_generate_loop.md)。）

---

## 十一、想自己写训练循环？这些零件可以直接拆下来

这是本篇最有实操价值的一节。如果你决定不用 `Trainer`，**不必从零重写所有东西**——
`Trainer` 里大量组件是 `@staticmethod` 或者不依赖 `self` 的，可以直接调用。

### 11.1 可以直接复用的（不需要实例化 `Trainer`）

| 组件 | 位置 | 为什么能直接用 |
|------|------|---------------|
| `Trainer.get_optimizer_cls_and_kwargs(args, model=None)` | `trainer.py:1278`（**`@staticmethod`**，装饰器在 `:1277`） | 完整实现了 `optim` 字符串 →(类, kwargs)，含 GaLore / LOMO / bitsandbytes 那些难搞的分支。内部只是查 `_OPTIMIZER_HANDLERS`（`trainer_optimizer.py:596`） |
| `Trainer.get_decay_parameter_names(model)` | `trainer.py:1308`（实例方法，但只读 `model`） | 返回「该加 weight_decay 的参数名」——自动排除 bias 和所有 norm 层 |
| `get_parameter_names(model, forbidden_layer_types, forbidden_layer_names)` | `trainer_pt_utils.py:1032` | 上一行的底层实现，纯函数 |
| `get_scheduler(name, optimizer, num_warmup_steps, num_training_steps, ...)` | `optimization.py:960` | 12 种 scheduler 的统一入口，纯函数 |
| `args.get_warmup_steps(num_training_steps)` | `training_args.py:2115` | 处理「`warmup_steps < 1` 当比例」的逻辑 |
| `LabelSmoother(epsilon=...)` | `trainer_pt_utils.py:436` | label smoothing 完整实现，`__call__(outputs, labels, shift_labels=...)` |
| `TrainerState` / `TrainerControl` / `CallbackHandler` / `ExportableState` | `trainer_callback.py:35` / `:234` / `:429` / `:189` | 状态与回调机制和训练循环解耦；`CallbackHandler` 只要一个回调列表就能用 |
| `DEFAULT_CALLBACKS` / `DEFAULT_PROGRESS_CALLBACK` | `trainer.py:184` / `:185`（notebook 分支 `:190`） | 回调集合的默认值 |
| `LengthGroupedSampler` / `DistributedLengthGroupedSampler` / `DistributedSamplerWithLoop` | `trainer_pt_utils.py:520` / `:563` / `:325` | 按长度分桶 + 分布式补齐，手写很烦 |
| `nested_gather` / `nested_concat` / `nested_detach` / `nested_numpify` | `trainer_pt_utils.py:219` / `:117` / `:174` / `:158` | 嵌套结构的分布式汇总，`evaluation_loop` 全靠它们 |
| `find_executable_batch_size(fn, batch_size, auto)` | `trainer_utils.py:938` | OOM 时自动降 batch 重跑（指数衰减），`train()` 就是把它当装饰器用的（`trainer.py:1452`） |
| `PREFIX_CHECKPOINT_DIR` / `get_last_checkpoint` / `sort_checkpoints` / `rotate_checkpoints` | `trainer_utils.py:266` / `:270` / `:282` / `:342` | 检查点目录约定与轮转 |
| `get_model_param_count` | `trainer_pt_utils.py:1015` | 数参数量（含 trainable_only 选项） |

### 11.2 必须自己实现的（`Trainer` 用 `self` 把它们缝在一起了）

| 自己要做的事 | `Trainer` 的对应实现 | 难点在哪 |
|-------------|---------------------|---------|
| **累积边界的同步控制** | `_set_sync_gradients`（`:1761`）+ `no_sync`（`:1767`） | 只在「窗口末尾」做 allreduce，中间的 backward 必须 `no_sync`。写错就是 `gas` 倍的通信浪费 |
| **loss 归一化** | 除以 `current_gradient_accumulation_steps`（`:1969`）或交给模型的 `num_items_in_batch` | 见第六节，最后一个不完整窗口最容易算错 |
| **混合精度 + GradScaler** | 完全交给 accelerate（第七节） | 手写要处理 `scaler.scale(loss).backward()` / `unscale_` / `scaler.step` / `scaler.update`，且它和梯度裁剪的顺序有讲究 |
| **梯度裁剪** | `_clip_grad_norm`（`:2550`）→ `accelerator.clip_grad_norm_` | 必须在 `unscale_` **之后**、`optimizer.step()` **之前**。顺序错了裁的就是被 scale 过的梯度 |
| **梯度检查点** | `model.gradient_checkpointing_enable()`（`trainer.py:1408`） | 调用点在 `train()` 里；还要注意 `use_cache` 必须关掉（`Trainer.__init__:616-617`） |
| **恢复训练** | `_init_training_state`（`:1558`）+ `skip_first_batches`（`:1718`） | 要把 dataloader **快进**到断点位置（`:1718`），还要同步 RNG 状态（`:1723` / `:1749`），否则 shuffle 顺序对不上 |
| **评估循环的分布式聚合** | `evaluation_loop`（`:2668`） | padding 对齐（`pad_across_processes`）、去重（`gather_for_metrics`）、`num_samples` 统计 |
| **回调状态落盘** | `ExportableState`（`trainer_callback.py:189`）+ `TrainerState.save_to_json`（`:143`） | 早停状态要能从 `trainer_state.json` 读回来 |

**最小可用配方**（把上面的零件拼起来）：

```python
# 1) 优化器：直接用 Trainer 的查表逻辑
opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(args, model)
decay = Trainer.get_decay_parameter_names(model)
groups = [
    {"params": [p for n, p in model.named_parameters() if n in decay and p.requires_grad],
     "weight_decay": args.weight_decay},
    {"params": [p for n, p in model.named_parameters() if n not in decay and p.requires_grad],
     "weight_decay": 0.0},
]
optimizer = opt_cls(groups, **opt_kwargs)          # 参数分组的细节见 create_optimizer，trainer.py:1184-1197

# 2) scheduler（含 warmup 的比例语义）
scheduler = get_scheduler(
    args.lr_scheduler_type, optimizer=optimizer,
    num_warmup_steps=args.get_warmup_steps(max_steps),
    num_training_steps=max_steps,
)
```

剩下必须自己写的核心就是这 10 行：

```python
for step, batch in enumerate(loader):
    with accelerator.accumulate(model):            # 或手写 no_sync + _set_sync_gradients
        with accelerator.autocast():               # 或交给 prepare_model 自动包
            loss = model(**batch)["loss"] / args.gradient_accumulation_steps
        accelerator.backward(loss)
        if accelerator.sync_gradients:             # 累积边界
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

**注意最后这段和 `Trainer` 的三个差异**：
① `Trainer` 是**先 clip 再算 grad_norm**（`:1796-1797`），且用 `float("inf")` 额外算一次原始范数；
② `Trainer` 在 `optimizer.step()` 之后**先读 lr 再 step scheduler**（`:1804` / `:1809`），
这样 log 里记的是「这一步用的 lr」；
③ `ReduceLROnPlateau` / `GreedyLR` **不在循环里 step**，而是等评估出指标后再 step（见 9.2）。

---

## 一句话总结

> **`Trainer` 那 4477 行里，真正每次都跑的是 `_run_epoch`（`trainer.py:1693`）里的三层 `for` 循环**：
> 外层 `update_step` 预取 `gradient_accumulation_steps` 个 batch，
> 内层逐 micro-batch 调 `training_step`（`:1907`）算 loss + `accelerator.backward`，
> 到累积边界才 `optimizer.step()` / `scheduler.step()` / `zero_grad()`（`:1800-1811`）。
>
> **「何时 log / eval / save / 停」不在循环里**，而在 `DefaultFlowCallback.on_step_end`（`trainer_callback.py:568`）
> 通过 `TrainerControl` 的 5 个布尔开关反向控制循环。
>
> 混合精度、分布式、梯度累积的**机制**全在 `accelerate` 手里，`Trainer` 只负责调度——
> 所以在 `Trainer` 里找不到 `torch.cuda.amp.autocast`，那个函数是个空壳（`trainer.py:2081-2086`）。

---

## 验证清单

以下命令均在仓库根目录实际执行过，输出即文中所引。

```bash
# 1. 文件规模
wc -l src/transformers/trainer.py src/transformers/training_args.py \
      src/transformers/trainer_callback.py src/transformers/trainer_pt_utils.py \
      src/transformers/trainer_optimizer.py src/transformers/trainer_seq2seq.py
# → 4477 / 2913 / 784 / 1342 / 615 / 392

# 2. TrainingArguments 的字段数（不是 200，是 112）
awk 'NR>=180 && NR<=2890 && /^    [a-zA-Z_][a-zA-Z_0-9]*: .*= field\(/ {c++} END {print c}' \
    src/transformers/training_args.py
# → 112

# 3. 主流程各入口行号（注意 save_model 在 _save_checkpoint 之后）
grep -n "def train(\|def _inner_training_loop\|def _run_epoch\|def training_step\|def compute_loss(\|def evaluate(\|def evaluation_loop\|def prediction_step\|def _save_checkpoint\|def save_model" \
    src/transformers/trainer.py
# → 1350 / 1465 / 1693 / 1907 / 1980 / 2569 / 2668 / 2936 / 3096 / 3811

# 4. 三层循环与关键动作的行号
grep -n "for epoch in range(epochs_trained, num_train_epochs)\|for update_step in range(\|for i, inputs in enumerate(batch_samples)\|do_sync_step = \|self.optimizer.step()\|self.lr_scheduler.step()\|model.zero_grad()\|loss = loss / self.current_gradient_accumulation_steps\|self.accelerator.backward(loss\|tr_loss_step = self.training_step" \
    src/transformers/trainer.py
# → 1538 / 1737 / 1755 / 1757 / 1800 / 1809 / 1811 / 1969 / 1976 / 1774

# 5. training_step 体内没有 optimizer.step / zero_grad（确认它是 forward_backward step）
sed -n '1907,1979p' src/transformers/trainer.py | grep -n "optimizer.step\|zero_grad"
# → 无匹配（grep 退出码 1）

# 6. autocast 是空壳
grep -n "autocast" src/transformers/trainer.py
# → 2075 / 2076 / 2077 / 2081 / 2083 / 2084
sed -n '2084p;2086p' src/transformers/trainer.py
# → "arguments, depending on the situation. We rely on accelerate for autocast, hence we do nothing here."
# → "        return contextlib.nullcontext()"

# 7. mixed_precision 的传递链
grep -n "mixed_precision" src/transformers/training_args.py | head -3
# → 1575（读环境变量）/ 1577（fp16 覆盖）/ 1579（bf16 覆盖）
grep -n '"mixed_precision"' src/transformers/trainer.py
# → 715（_build_accelerator_args 里，交给 Accelerator）

# 8. ForCausalLMLoss 的 shift + cross_entropy
grep -n "" src/transformers/loss/loss_utils.py | sed -n '32p;39p;45p;49p;59p;63p;64p;67p;70p;71p'
# → 32:def fixed_cross_entropy(  / 39:reduction = "sum" if ... / 45:loss = loss / num_items_in_batch
# → 49:def ForCausalLMLoss(      / 59:logits = logits.float()   / 63:nn.functional.pad(labels, (0, 1), ...)
# → 64:shift_labels = labels[..., 1:].contiguous()  / 67:logits = logits.view(-1, vocab_size)
# → 70:loss = fixed_cross_entropy(...)  / 71:return loss

# 9. 检查点常量与目录名
sed -n '243,248p' src/transformers/trainer.py
# → TRAINING_ARGS_NAME / TRAINER_STATE_NAME / OPTIMIZER_NAME / SCALER_NAME / OPTIMIZER_NAME_BIN / SCHEDULER_NAME
grep -n "PREFIX_CHECKPOINT_DIR = " src/transformers/trainer_utils.py
# → 266:PREFIX_CHECKPOINT_DIR = "checkpoint"
grep -n "checkpoint_folder = f" src/transformers/trainer.py
# → 3103

# 10. 回调类行号与默认回调
grep -n "^class TrainerState\|^class ExportableState\|^class TrainerControl\|^class TrainerCallback\|^class CallbackHandler\|^class DefaultFlowCallback\|^class ProgressCallback\|^class PrinterCallback\|^class EarlyStoppingCallback" \
    src/transformers/trainer_callback.py
# → 35 / 189 / 234 / 295 / 429 / 563 / 624 / 698 / 711
grep -n "DEFAULT_CALLBACKS = \|DEFAULT_PROGRESS_CALLBACK = " src/transformers/trainer.py
# → 184 / 185 / 190

# 11. 回调在 __init__ 里的装配与复位规则
sed -n '568p;578p;581p;613p' src/transformers/trainer.py
sed -n '504,508p' src/transformers/trainer_callback.py     # on_step_begin 清三个开关
sed -n '443p' src/transformers/trainer_callback.py         # DefaultFlowCallback 缺失告警

# 12. 复用零件的位置
grep -n "def get_optimizer_cls_and_kwargs\|def get_decay_parameter_names\|def create_optimizer\|def create_scheduler" \
    src/transformers/trainer.py
# → 1171(create_optimizer) / 1247(create_scheduler) / 1278(get_optimizer_cls_and_kwargs) / 1308(get_decay_parameter_names)
sed -n '1277p' src/transformers/trainer.py                          # → "    @staticmethod"
grep -n "^def get_scheduler" src/transformers/optimization.py          # → 960
grep -n "^def get_parameter_names" src/transformers/trainer_pt_utils.py # → 1032
grep -n "^def find_executable_batch_size" src/transformers/trainer_utils.py # → 938
grep -n "^_OPTIMIZER_HANDLERS" src/transformers/trainer_optimizer.py    # → 596

# 13. Llama 的 forward 有 **kwargs ⇒ model_accepts_loss_kwargs=True ⇒ 走 num_items_in_batch 新路径
grep -n "Unpack\[TransformersKwargs\]" src/transformers/models/llama/modeling_llama.py | tail -1
# → 448（LlamaForCausalLM.forward）
grep -n "model_accepts_loss_kwargs = any" -A 2 src/transformers/trainer.py
# → 503-505（inspect.signature 找 VAR_KEYWORD）

# 14. 不要跑的（会改写仓库）
# ❌ make fix-repo  ❌ python utils/modular_model_converter.py
```

**未能本地验证的一项（如实标注）**：第七节里「accelerate 在 `prepare_model()` 时把 `model.forward`
包进 autocast」这一步属于 **accelerate 内部实现**，
本机未安装 `accelerate`（`pip show accelerate` → `Package(s) not found`），
无法在本仓库内验证。**本仓库内可验证的部分**只有：`mixed_precision` 的推导（`training_args.py:1575-1579`）、
传给 `Accelerator`（`trainer.py:715`）、以及 `Trainer` 侧的 autocast 是空壳（`trainer.py:2081-2086`）这三条。

---

## 导航

- 路线图：[00_roadmap.md](./00_roadmap.md)（本篇 = 入门 · 架构 · 1A6）
- 上一篇：[1A5_generate_loop.md](./1A5_generate_loop.md) —— `generate` 主循环
- 下一篇：[2A1_attention_backends.md](./2A1_attention_backends.md) —— 注意力后端机制
  （1A6 是「入门 · 架构」的最后一篇，下一篇跨到「进阶 · 架构」）
- **建议紧接着读**：[2A5_modular_code_generation.md](./2A5_modular_code_generation.md) ——
  第五节里 `model.loss_function` / `LOSS_MAPPING` 的机制，正好用 2A5 的视角看一遍：
  生成文件里那个 `self.loss_function(...)` 调用是怎么被 converter 从父模型抄过来的。
