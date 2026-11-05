# 同学 A：雷达记录上的跌倒识别

这个目录在电脑上运行，不参与 WS63 固件编译。

默认用电脑端滑窗分类器：I/Q 幅度为主，`gear`/`rng`/`vel`/`snr` 为辅，七类动作。看到 `fall` 之后仍用原来的静止 3 秒疑似、8 秒报警。`params.json` 里 `use_classifier` 为 false 时回退旧规则。合成数据只用来把训练和事件接口跑通，不是实板准确率。

人短暂消失和无效帧一样，不把这段空档算进静止或活动计时；冲击已开始或已经报警时，丢人不复位。报警只能靠同学 B 回传的 `{"cmd":"clear"}` 解除，由 `apply_uplink` 接收。评测看记录结束时的状态。

`vel_fall` 没有放宽。缓慢倒下和主动躺下都到不了这个速度，放宽会一起报警。没有高度、距离也不回升时，快速坐下和快走急停仍会报警。

实板单位还没有确认。合成数据按厘米、厘米每秒生成，并带了合成高度，用来检查文件格式、事件格式和程序能不能跑完。它不是实板准确率。检测不用 `snr` 和 `gear`。

## 运行

在 `src` 目录：

```
python -m fall_recog run
```

常用命令：

```
python -m fall_recog check fall_recog/data/synthetic/test
python -m fall_recog detect fall_recog/data/synthetic/test --split test
python -m fall_recog eval --input fall_recog/data/board --out fall_recog/results/board --title 实板测试
```

实板 CSV 放到 `fall_recog/data/board`，命名和列见 `docs/数据接口.md`。确认单位后只改 `params.json` 里的数字。

## 接口

- 和雷达采集同学：`docs/数据接口.md`，采集步骤在 `docs/采集标注规范.md`。
- 和同学 B：`docs/事件接口.md`。正常、疑似、报警三条状态，外加解除命令。

## 报告

`python -m fall_recog run` 会重写这些文件：

- `results/测试结果_test.md`：P01，规范动作
- `results/测试结果_train.md`：P02、P03，同一套规则，不参与改门限
- `results/规则边界.md`：有高度时快速坐下和急停站立不报警；缓慢倒下仍不报警
- `results/event_example.jsonl`：一段跌倒的事件序列
