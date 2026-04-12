
<requirement>

目前有两个agent：

- 一个是src/agent/code_agent.py，实现的是最基础版本的ReAct Agent
- 一个是src/agent/code_agent_with_reflection.py，实现的是继承了self-reflection功能的Agent

他们都在src/benchmarks/swe_bench_runner.py这个地方被调用。

现在，我想实现一个更高级的code agent，名为ForwardContextManagement Agent，本质上是为了减少token。

**我先补充一些背景，方案你了解：**

当前ReAct-based的agent成本很高，原因在于：
- 每次agent交互的时候，observation有时会很长。(如execution log，读取大文件)
- 复杂任务需要agent多轮交互。轮次变多时，token消耗爆炸增长（尽管考虑KV Cache）

目前的方法可以总结为Backward的context管理。

```
Observation角度: 基于规则，省略最早k轮的observation。
- 有篇实验论文[2]表明，直接Mask掉observation可能比简单的LLM summarize效果更好
DeepMiner(Search Agent) [4]为了训练和推理能进行多轮次，mask掉最早的k轮observation，但是保留think和action
Thought & Observation角度: 
- [6] 利用便宜的LLM（Gemini-flash）从定义的三个维度（无用信息，冗余信息，过期信息）来对历史trajectory进行总结。技术角度，他设计了slide window，可以选择性地总结
```

但是，存在下面问题，导致成本优化不明显：

- Backward的context管理会让Cache机制失灵。
- 忽略Action，agent交互轮次无法减少。

我们想做一种Backward的context管理。Backward的context管理的本质是确保当前步骤的简洁和正确。不仅仅简化Thought和Observation，还会调整Action。而调整A需要我们往前看k步，这需要一定的推理能力，不能用简单的heuristic的方法。
因此，我们提出了第一个面向向前context管理的agent，可以优化成本，提高精度。

**ForwardContextManagement Agent**的设计思路是这样的：

包含一个ReAct-based Code Agent Agent_code 用来一步步完成SWE的任务。就像src/agent/code_agent.py的一样
和
一个ReAct-based ForwardContextManagement Agent Agent_context 用来确定每次A_code生成的action，和对thought和observation进行summarize，从而保证每一步都是简洁且正确的。
这样做的好处是不会破坏KV cache，并且轮次会大大减少，因为action确保正确。

你需要先定义prompt来定义他们要做的事情，prompt的设计还要参考他们的工作流程，prompt的结构可以参考：src/prompts/react_baseline.j2。

他们的工作流程是这样的：

对于输入的prompt和question，Agent_code 先生成thinking content，为T_0，和对应的Action A_0，和执行得到的Observation O_0。所有这些生成的内容，和之前的交互信息（当前步骤还没实现）都会作为输入给到A_context agent，他首先会生成thought summary T'_0，再观察A_0是否正确，如果需要进一步和环境交互才能判断是否正确，可以让A_context一步步和环境交互，比如说读取大文件，大的log记录等，实现的架构也是ReAct-based Agent去验证Action A_0的正确与否，如果根据和环境交互的过程发现Action A_0不正确，那么让这个agent生成正确的A'_0，如果A_0正确，则直接不修改，A'_0=A_0。最后得到T'_0和A'_0，并执行得到Observation O'_0，A_context再对O'_0进行summarize，得到O''_0，这就是一步的完整过程，再进入下一步。

具体实现层面，你可以封装成一个condenser。我举个例子：

考虑一个这样的trajectory：(T0,A0,O0), (T1,A1,O1), ..., (Ti,Ai,Oi) ---> (Ti+1,Ai+1,Oi+1)
即当前输入到第i步，生成i+1步的内容

之前的condenser是这样做的：将(T0,A0,O0), (T1,A1,O1), ..., (Ti,Ai,Oi)变成简洁的(T0,A0,O0), (T1,A1,O1), ..., (Tk,Ak,Ok)，他修改了前面k步的context长度，同时破坏了kv cache。

我想做一个这样的condenser：

我们对之前生成的0～i-1步直接复制，保持不变，对第i步(Ti,Ai,Oi)进行优化。先根据Oi判断Ai是否正确，如果正确，则对T_i和O_0进行summarize即可。如果不正确或者需要进一步判断，则调用一个Agent_context和环境进行交互，并最终输出一个正确的action A'i，和执行结果 O'i，并对Ti进行summarize和补充，得到T'i。最终得到(T'i,A'i,O'i)，所以condenser的输出就是(T0,A0,O0), (T1,A1,O1), ..., (Ti-1,Ai-1,Oi-1),(T'i,A'i,O'i)

在prompt中告诉Agent_context，不要执行破坏性action，如果必须要，确保好备份。

</requirement>

根据上述要求，在具体实现过程中，你要做这些事情:

- 修改src/prompts/react_baseline.j2的文件名，这个文件记录了baseline React-based agent的prompt，但是现在文件名看不出来。并修改所有调用了这个文件的代码
- 在src/prompts中添加满足上述要求的agent对应的prompt
- 参考src/agent/definition/context_condenser.md构建满足上述要求的condenser
- 修改src/benchmarks/swe_bench_runner.py的文件，确保能成功调用。