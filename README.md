# Review-Excellent-AI-content

Latest book which I recently read was very excellent was on RL (https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)

Here the main topics which is discussed about:
1. Introduction to Reinforcement Learning
1.1. Definition and Scope
Reinforcement Learning (RL) is a computational approach to understanding and automating goal-directed learning and decision-making. It is distinguished from other machine learning paradigms by its emphasis on learning from interaction to achieve long-term objectives.

Agent and Environment: The agent is the learner or decision-maker, and the environment is everything the agent interacts with. The agent's actions affect the state of the environment, and in turn, the environment provides observations and rewards to the agent.
1.2. Elements of Reinforcement Learning
RL involves several core elements:

Policy (
π
π): A policy is a mapping from states to actions. It defines the agent's behavior at a given time. Policies can be deterministic (
π
(
s
)
=
a
π(s)=a) or stochastic (
π
(
a
∣
s
)
=
P
[
A
t
=
a
∣
S
t
=
s
]
π(a∣s)=P[A 
t
​
 =a∣S 
t
​
 =s]).

Reward Signal (R): The reward is a scalar feedback signal indicating how well the agent is performing at a given time. The agent's objective is to maximize the cumulative reward.

Value Function (V): The value function estimates the expected return from a state (or state-action pair), indicating the long-term desirability of states.

Model of the Environment: In model-based RL, the agent has a model that mimics the behavior of the environment. Models can be used for planning by simulating future states and rewards.

1.3. Goals of Reinforcement Learning
The primary goal of RL is to find an optimal policy that maximizes the expected cumulative reward over time. This involves balancing exploration (trying new actions to discover their effects) and exploitation (using known actions that yield high rewards).

2. Markov Decision Processes (MDPs)
2.1. Formal Definition
An MDP provides a mathematical framework for modeling sequential decision-making problems where outcomes are partly under the control of the agent and partly random.

An MDP is defined by the tuple 
(
S
,
A
,
P
,
R
,
γ
)
(S,A,P,R,γ), where:

S
S: A finite set of states.
A
A: A finite set of actions.
P
(
s
′
,
r
∣
s
,
a
)
P(s 
′
 ,r∣s,a): Transition probability distribution, the probability of moving to state 
s
′
s 
′
  and receiving reward 
r
r given current state 
s
s and action 
a
a.
R
(
s
,
a
)
R(s,a): Expected reward received after transitioning from state 
s
s using action 
a
a.
γ
∈
[
0
,
1
]
γ∈[0,1]: Discount factor, representing the difference in importance between future and immediate rewards.
2.2. Markov Property
The Markov property states that the future is independent of the past given the present state. Formally, 
P
[
S
t
+
1
∣
S
t
]
=
P
[
S
t
+
1
∣
S
1
,
S
2
,
.
.
.
,
S
t
]
P[S 
t+1
​
 ∣S 
t
​
 ]=P[S 
t+1
​
 ∣S 
1
​
 ,S 
2
​
 ,...,S 
t
​
 ].

2.3. Return and Value Functions
Return (
G
t
G 
t
​
 ): The total discounted return from time 
t
t is 
G
t
=
∑
k
=
0
∞
γ
k
R
t
+
k
+
1
G 
t
​
 =∑ 
k=0
∞
​
 γ 
k
 R 
t+k+1
​
 .

State-Value Function (
V
π
(
s
)
V 
π
 (s)): The expected return when starting in state 
s
s and following policy 
π
π:

V
π
(
s
)
=
E
π
[
G
t
∣
S
t
=
s
]
.
V 
π
 (s)=E 
π
​
 [G 
t
​
 ∣S 
t
​
 =s].
Action-Value Function (
Q
π
(
s
,
a
)
Q 
π
 (s,a)): The expected return when starting from state 
s
s, taking action 
a
a, and thereafter following policy 
π
π:

Q
π
(
s
,
a
)
=
E
π
[
G
t
∣
S
t
=
s
,
A
t
=
a
]
.
Q 
π
 (s,a)=E 
π
​
 [G 
t
​
 ∣S 
t
​
 =s,A 
t
​
 =a].
2.4. Optimality
Optimal Policy (
π
∗
π 
∗
 ): A policy is optimal if it yields the highest value function 
V
∗
(
s
)
=
max
⁡
π
V
π
(
s
)
V 
∗
 (s)=max 
π
​
 V 
π
 (s) for all 
s
∈
S
s∈S.

Bellman Optimality Equations:

For state-value function:
V
∗
(
s
)
=
max
⁡
a
∑
s
′
,
r
P
(
s
′
,
r
∣
s
,
a
)
[
r
+
γ
V
∗
(
s
′
)
]
.
V 
∗
 (s)= 
a
max
​
  
s 
′
 ,r
∑
​
 P(s 
′
 ,r∣s,a)[r+γV 
∗
 (s 
′
 )].
For action-value function:
Q
∗
(
s
,
a
)
=
∑
s
′
,
r
P
(
s
′
,
r
∣
s
,
a
)
[
r
+
γ
max
⁡
a
′
Q
∗
(
s
′
,
a
′
)
]
.
Q 
∗
 (s,a)= 
s 
′
 ,r
∑
​
 P(s 
′
 ,r∣s,a)[r+γ 
a 
′
 
max
​
 Q 
∗
 (s 
′
 ,a 
′
 )].
3. Dynamic Programming (DP)
3.1. Overview
Dynamic Programming refers to a collection of algorithms that can be used to compute optimal policies given a perfect model of the environment as an MDP.

3.2. Policy Evaluation
Policy Evaluation aims to compute the value function 
V
π
(
s
)
V 
π
 (s) for a given policy 
π
π. This is done by solving the Bellman Expectation Equation:

V
π
(
s
)
=
∑
a
π
(
a
∣
s
)
∑
s
′
,
r
P
(
s
′
,
r
∣
s
,
a
)
[
r
+
γ
V
π
(
s
′
)
]
.
V 
π
 (s)= 
a
∑
​
 π(a∣s) 
s 
′
 ,r
∑
​
 P(s 
′
 ,r∣s,a)[r+γV 
π
 (s 
′
 )].
This can be solved iteratively using Iterative Policy Evaluation:

Initialize 
V
(
s
)
V(s) arbitrarily.
Update:
V
k
+
1
(
s
)
=
∑
a
π
(
a
∣
s
)
∑
s
′
,
r
P
(
s
′
,
r
∣
s
,
a
)
[
r
+
γ
V
k
(
s
′
)
]
.
V 
k+1
​
 (s)= 
a
∑
​
 π(a∣s) 
s 
′
 ,r
∑
​
 P(s 
′
 ,r∣s,a)[r+γV 
k
​
 (s 
′
 )].
Repeat until convergence.
3.3. Policy Improvement
Policy Improvement involves improving the policy based on the current value function. The Policy Improvement Theorem states that a policy 
π
′
π 
′
  is better than or equal to policy 
π
π if:

π
′
(
s
)
=
arg
⁡
max
⁡
a
∑
s
′
,
r
P
(
s
′
,
r
∣
s
,
a
)
[
r
+
γ
V
π
(
s
′
)
]
.
π 
′
 (s)=arg 
a
max
​
  
s 
′
 ,r
∑
​
 P(s 
′
 ,r∣s,a)[r+γV 
π
 (s 
′
 )].
3.4. Policy Iteration
Policy Iteration alternates between policy evaluation and policy improvement until the policy converges to the optimal policy.

Policy Evaluation: Compute 
V
π
V 
π
  for the current policy 
π
π.
Policy Improvement: Generate a new policy 
π
′
π 
′
  using the current value function.
Repeat steps 1 and 2 until 
π
π does not change.
3.5. Value Iteration
Value Iteration combines policy evaluation and improvement into a single step by iteratively applying the Bellman Optimality Operator:

V
k
+
1
(
s
)
=
max
⁡
a
∑
s
′
,
r
P
(
s
′
,
r
∣
s
,
a
)
[
r
+
γ
V
k
(
s
′
)
]
.
V 
k+1
​
 (s)= 
a
max
​
  
s 
′
 ,r
∑
​
 P(s 
′
 ,r∣s,a)[r+γV 
k
​
 (s 
′
 )].
Value iteration converges to 
V
∗
V 
∗
 , from which the optimal policy can be derived.

3.6. Asynchronous Dynamic Programming
Asynchronous DP algorithms update the value of states in any order, potentially leading to faster convergence in practice.

4. Monte Carlo Methods
4.1. Overview
Monte Carlo (MC) methods learn from complete episodes by averaging sample returns. They are suitable for episodic tasks and do not require knowledge of the environment's dynamics.

4.2. Estimating Value Functions
For a policy 
π
π, the MC method estimates 
V
π
(
s
)
V 
π
 (s) as the average of returns following visits to state 
s
s.

First-Visit MC: Estimates 
V
π
(
s
)
V 
π
 (s) as the average of returns following the first visit to 
s
s in each episode.
Every-Visit MC: Estimates 
V
π
(
s
)
V 
π
 (s) as the average of returns following every visit to 
s
s.
4.3. Monte Carlo Control
MC control methods aim to find the optimal policy by iteratively improving the policy based on MC estimates.

Exploring Starts: Ensures that all state-action pairs have a non-zero probability of being sampled.
ϵ
ϵ-Greedy Policies: Introduces exploration by choosing a random action with probability 
ϵ
ϵ and the best-known action with probability 
1
−
ϵ
1−ϵ.
4.4. Off-Policy Prediction and Control
Importance Sampling: Used to correct the distribution of returns when following a behavior policy different from the target policy.
Off-Policy MC: Estimates 
V
π
V 
π
  while following a different behavior policy 
b
b.
5. Temporal-Difference (TD) Learning
5.1. TD Prediction
TD learning combines ideas from DP and MC methods. It updates value estimates based on other learned estimates without waiting for the final outcome.

TD(0) Update Rule:
V
(
S
t
)
←
V
(
S
t
)
+
α
[
R
t
+
1
+
γ
V
(
S
t
+
1
)
−
V
(
S
t
)
]
.
V(S 
t
​
 )←V(S 
t
​
 )+α[R 
t+1
​
 +γV(S 
t+1
​
 )−V(S 
t
​
 )].
The term 
[
R
t
+
1
+
γ
V
(
S
t
+
1
)
−
V
(
S
t
)
]
[R 
t+1
​
 +γV(S 
t+1
​
 )−V(S 
t
​
 )] is the TD error.
5.2. Advantages of TD Learning
Sample Efficiency: Learns from incomplete episodes.
Bootstraping: Updates estimates based on existing estimates.
5.3. TD Control
Extends TD prediction to learn optimal policies.

SARSA (On-Policy TD Control):

Q
(
S
t
,
A
t
)
←
Q
(
S
t
,
A
t
)
+
α
[
R
t
+
1
+
γ
Q
(
S
t
+
1
,
A
t
+
1
)
−
Q
(
S
t
,
A
t
)
]
.
Q(S 
t
​
 ,A 
t
​
 )←Q(S 
t
​
 ,A 
t
​
 )+α[R 
t+1
​
 +γQ(S 
t+1
​
 ,A 
t+1
​
 )−Q(S 
t
​
 ,A 
t
​
 )].
Q-Learning (Off-Policy TD Control):

Q
(
S
t
,
A
t
)
←
Q
(
S
t
,
A
t
)
+
α
[
R
t
+
1
+
γ
max
⁡
a
Q
(
S
t
+
1
,
a
)
−
Q
(
S
t
,
A
t
)
]
.
Q(S 
t
​
 ,A 
t
​
 )←Q(S 
t
​
 ,A 
t
​
 )+α[R 
t+1
​
 +γ 
a
max
​
 Q(S 
t+1
​
 ,a)−Q(S 
t
​
 ,A 
t
​
 )].
5.4. Expected SARSA
Computes the expected value over all possible actions under the current policy:

Q
(
S
t
,
A
t
)
←
Q
(
S
t
,
A
t
)
+
α
[
R
t
+
1
+
γ
∑
a
π
(
a
∣
S
t
+
1
)
Q
(
S
t
+
1
,
a
)
−
Q
(
S
t
,
A
t
)
]
.
Q(S 
t
​
 ,A 
t
​
 )←Q(S 
t
​
 ,A 
t
​
 )+α[R 
t+1
​
 +γ 
a
∑
​
 π(a∣S 
t+1
​
 )Q(S 
t+1
​
 ,a)−Q(S 
t
​
 ,A 
t
​
 )].
6. Eligibility Traces
6.1. Concept
Eligibility traces provide a bridge between MC and TD methods, allowing the algorithm to assign credit to previous states and actions for future rewards.

6.2. TD(
λ
λ)
Combines TD(0) and MC methods using the parameter 
λ
∈
[
0
,
1
]
λ∈[0,1].

Forward View: The update is based on the 
λ
λ-return, which is a weighted average of n-step returns.
Backward View: Implements eligibility traces that decay over time:
e
t
(
s
)
=
γ
λ
e
t
−
1
(
s
)
+
I
[
S
t
=
s
]
.
e 
t
​
 (s)=γλe 
t−1
​
 (s)+I[S 
t
​
 =s].
The update rule is:
V
(
s
)
←
V
(
s
)
+
α
δ
t
e
t
(
s
)
,
V(s)←V(s)+αδ 
t
​
 e 
t
​
 (s),
where 
δ
t
δ 
t
​
  is the TD error.
6.3. Advantages
Bias-Variance Trade-off: The 
λ
λ parameter controls the trade-off between bias and variance.
Faster Learning: Eligibility traces can lead to faster convergence by updating multiple states per time step.
7. Function Approximation
7.1. Need for Function Approximation
In problems with large or continuous state spaces, it's impractical to maintain value estimates for every possible state. Function approximation generalizes learning to unseen states.

7.2. Types of Function Approximators
Linear Function Approximation:

V
^
(
s
,
w
)
=
w
⊤
ϕ
(
s
)
,
V
^
 (s,w)=w 
⊤
 ϕ(s),
where 
ϕ
(
s
)
ϕ(s) is a feature vector.

Non-Linear Function Approximation: Uses neural networks or other non-linear models to approximate the value function.

7.3. Gradient Descent Methods
Adjust the weights 
w
w to minimize the Mean Squared Error (MSE) between the estimated value and the target.

Stochastic Gradient Descent Update:
w
←
w
+
α
[
R
t
+
1
+
γ
V
^
(
S
t
+
1
,
w
)
−
V
^
(
S
t
,
w
)
]
∇
w
V
^
(
S
t
,
w
)
.
w←w+α[R 
t+1
​
 +γ 
V
^
 (S 
t+1
​
 ,w)− 
V
^
 (S 
t
​
 ,w)]∇ 
w
​
  
V
^
 (S 
t
​
 ,w).
7.4. Compatible Function Approximation
For policy gradient methods, the function approximator must be compatible with the policy to ensure convergence.

8. Policy Gradient Methods
8.1. Overview
Policy gradient methods optimize the policy directly by adjusting parameters to maximize expected reward.

8.2. Policy Parameterization
The policy is parameterized by 
θ
θ:

Stochastic Policies:
π
(
a
∣
s
,
θ
)
=
P
[
A
t
=
a
∣
S
t
=
s
,
θ
]
.
π(a∣s,θ)=P[A 
t
​
 =a∣S 
t
​
 =s,θ].
8.3. Objective Function
The goal is to maximize the expected return:

J
(
θ
)
=
E
π
[
G
t
]
.
J(θ)=E 
π
​
 [G 
t
​
 ].
8.4. Policy Gradient Theorem
The gradient of the expected return with respect to the policy parameters is:

∇
J
(
θ
)
=
E
π
[
∇
ln
⁡
π
(
A
t
∣
S
t
,
θ
)
Q
π
(
S
t
,
A
t
)
]
.
∇J(θ)=E 
π
​
 [∇lnπ(A 
t
​
 ∣S 
t
​
 ,θ)Q 
π
 (S 
t
​
 ,A 
t
​
 )].
8.5. REINFORCE Algorithm
A Monte Carlo policy gradient method:

Generate episodes using the current policy 
π
π.
For each time step 
t
t, update the policy parameters:
θ
←
θ
+
α
[
G
t
−
b
(
S
t
)
]
∇
θ
ln
⁡
π
(
A
t
∣
S
t
,
θ
)
,
θ←θ+α[G 
t
​
 −b(S 
t
​
 )]∇ 
θ
​
 lnπ(A 
t
​
 ∣S 
t
​
 ,θ),
where 
b
(
S
t
)
b(S 
t
​
 ) is a baseline to reduce variance.
8.6. Actor-Critic Methods
Combines policy gradient (actor) with value function estimation (critic):

Actor: Updates policy parameters in the direction suggested by the critic.
Critic: Estimates the value function 
V
π
(
s
)
V 
π
 (s) to evaluate the actor's performance.
8.7. Advantage Function
To reduce variance, use the advantage function 
A
π
(
s
,
a
)
=
Q
π
(
s
,
a
)
−
V
π
(
s
)
A 
π
 (s,a)=Q 
π
 (s,a)−V 
π
 (s) in place of 
Q
π
(
s
,
a
)
Q 
π
 (s,a).

9. Exploration and Exploitation
9.1. The Exploration-Exploitation Trade-off
Exploration: Trying new actions to discover their potential rewards.
Exploitation: Choosing the best-known action to maximize immediate reward.
9.2. Strategies for Exploration
ϵ
ϵ-Greedy Policy: With probability 
ϵ
ϵ, select a random action; otherwise, select the best-known action.
Softmax Action Selection: Select actions probabilistically according to their preference values.
9.3. Optimistic Initialization
Initialize value estimates optimistically to encourage exploration of less-visited states.

9.4. Upper Confidence Bound (UCB)
Select actions based on both estimated value and uncertainty:

A
t
=
arg
⁡
max
⁡
a
[
Q
(
s
,
a
)
+
c
ln
⁡
t
N
(
s
,
a
)
]
,
A 
t
​
 =arg 
a
max
​
 [Q(s,a)+c 
N(s,a)
lnt
​
 
​
 ],
where 
c
c controls the degree of exploration, and 
N
(
s
,
a
)
N(s,a) is the number of times action 
a
a has been selected in state 
s
s.
