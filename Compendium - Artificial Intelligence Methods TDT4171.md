# Artificial Intelligence Methods


## Chapter 13 - Quantifying Uncertainty


Agents need to handle uncertainty due to partial observability or nondeterminism.

With uncertainty you need to make the right decision - the rational decision. 

Probability to quantify uncertainty, we can say it is 80 % chance of rain. 

To deal with decisions with uncertainty you need to know preferences of the possible outcomes. We can use utility theory by stating the utilities (degree of usefullness) of each outcome.


**Decision theory = probability theory + utility theory**

Maximum expected utility (MEU), a rational agent chooses the action with highest expected utility.

In probability theory the set of all possible worlds is called the **sample set**.

#### Probability model

$$
\sum_{w \in \Omega} P(w) = 1
$$


Cases with a certain outcome is called events, and can be described by propositions in a formal language. 


Unconditional: $P(doubles)$
Conditional: $P(doubles | Die_1 = 5)$

#### Product rule

$$
P(a | b) = \frac{P(a \land b)}{P(b)}
$$

#### Random variables

**Domain:** $Die_1$ = {1, ..., 6}

* A boolean random variable has the domain *{true, false}*

We have:

* P(Weather = sunny) = 0.6
* P(Weather = rain) = 0.1
* P(Weather = cloudy) = 0.29
* P(Weather = snow) = 0.01

**P**(Weather) = (0.6, 0.1, 0.29, 0.01)

> We say that the P statement defines a probability distribution for the random variable $Weather$

**Probability density function**:

$P(Temp = x) = Uniform_{[18C, 26C]}(x)$

#### Joint probability distribution of Weather and Cavity

$P(Weather, Cavity) = P(Weather | Cavity) * P(Cavity)$

> A possible world is defined to be an assignment of values to all of the random variables under consideration. $\implies$ joint distribution for all of the random variables - the so-called full joint probability distribution. 


Rules:

* $P(a \lor b) = P(a) + P(b) - P(a \land b)$

**Wrong beliefs can result in bad decisions, even though they are rational, given the agents beliefs.**

#### Probabilistic interference


> Full joint distribution is a table stating all combinations with given probabilities.


### Independence

Since weather is independant of dental problems and vice versa, we have:

$P(Toothache, Catch, Cavity, Weather) = P(Toothache, Catch, Cavity) P(Weather)$ 

Absolute independence $\implies P(a \land b) = P(a) P(b)$

$P(C_1,...,C_n)$, has $2^n$ entries, but can be represented with $n$ single variable distributions $P(C_i)$.


### Bayes' rule

$$
P(b | a) = \frac{P(a | b) P(b)}{P(a)}
$$

, with background evidence **e** we have:

$$
P(Y | X, e) = \frac{P(X | Y, e) P(Y | e)}{P(X | e)}
$$

#### Conditional independence

$P(X,Y | Z) = P(X | Z) P(Y | Z)$

We have the example:

$$
P(Toothache, Catch, Cavity)
= P(Toothache, Catch | Cavity) P(Cavity)
= P(Tootache | Cavity) P(Catch | Cavity) P(Cavity)
$$

Generally,

$P(Cause, Effect_1,..., Effect_n) = P(Cause) \Pi_i P(Effect_i | Cause)$

This is called a *naive Bayes* model, and it is called naive because often the effect variables are not actually conditionally independent given the cause variable. 


## Chapter 14 - Probabilistic Reasoning

#### Bayesian network

Directed graph which each node is annotated with quantitive probability information:

1. Each node corresponds to a random variable, continuous or discrete
2. A set of directed links or arrows connects pairs of nodes. If there is an arrow from node X to node Y, X is said to be a parent of Y. The graph has no directed cycles (DAG).
3. Each node $X_i$ has a conditional probability distribution $P(X_i | Parents(X_i))$ that quantifies the effect of the parents on the node.


An arrow means that X has a direct influence on Y. 


**Simple bayesian network:**

(Weather)
(Cavity) --> (Toothache)
 |
 v
 (Catch)
 
 
**Conditional probability table (CPT):**

| B | E | P(A)|
|---|----|----|
| t | t | 0.95 |
| t | f | 0.92 |
| f | t | 0.90 |
| f | f | 0.001 |



We have that $P(A | B = t \land E = t) = 0.95$

You can look at Bayesian networks as an encoding of a collection of conditional independence statements. 

$$
P(x_1,..., x_n) = \Pi^{n}_{i=1} P(x_i | parents(X_i))
$$

For the example in the book, we have:

$P(MaryCalls | JohnCalls , Alarm, Earthquake, Burglary) = P(MaryCalls | Alarm)$.

### Chain rule

$$
P(x_1,..., x_n) = P(x_n | x_{n-1},...,x_1) P(x_{n-1} | x_{n-2},..., x_1) ... P(x_2 | x_1) P(x_1)
= \Pi^n_{i=1} P(x_i | x_{i-1},..., x_1)
$$


Constructing a Bayesian network:

* Choose the links between the nodes that *directly* influence the other.


 
#### Markov blanket

A node is conditionally independent of all other nodes in the network, given its parents, children and children's parents - that is, given its Markov blanket.

This means that the Markov blanket of a node is the only knowledge needed to predict the behavior of that node and its children.

If the maximum number of parents is *k*, filling in the CPT for a node requires up to $O(2^k)$ numbers.

##### Deterministic nodes

A deterministic node has its value specified exactly by the values of its parents, with no uncertainty. 


Uncertain relationships can often be characterized by so-called noisy logical relationships. the standard example is the **noisy-OR** relation, which is a generaliztion og the logical OR. The noisy OR model allows for uncertainty.

**Noisy-OR and inhibition nodes:**

* $q_{cold} = P(¬fever | cold,¬flu,¬malaria) = 0.6$
* $q_{flu} = P(¬fever | ¬cold, flu,¬malaria) = 0.2$ ,
* $q_{malaria} = P(¬fever | ¬cold,¬flu, malaria) = 0.1$

This gives:

$$
P(x_i | parents(X_i)) = 1 − \prod_{j:Xj =true} q_j
$$

#### Continious distributions

Example Gaussian distribution $N(μ, σ^2)(x)$, with mean μ and variance $σ^2$ as parameters.


### Interference in Bayesian Networks

The basic task for any prob. inference system is to compute posteriour probability distribution for a set of query variabels, given som observed event - that is, some assignment of values to a set of evidence variables. 

We will use the notation: X denotes the query variable; E denotes the set of evidence variables $E_1, ... E_m$ and *e* is a particular observed evet; Y will denote the nonevidence, nonquery variables $Y_1,..., Y_l$ (called hidden variables).

A typical query asks for the posterior probability distribution $P(X | e)$.

In the burglary network, we might observe the event in which $JohnCalls = true$ and $MaryCalls = true$. We could then ask for, say, the probability that a burglary has occurred:
$$
P(Burglary | JohnCalls=true, MaryCalls=true) = (0.284, 0.716)
$$

#### Inference by enumeration

Consider the query $P(Burglary | JohnCalls=true,MaryCalls=true)$. The hiddenvariables for this query are Earthquake and Alarm.

We get:

$P(B | j, m) = \alpha P(B,j,m) = \alpha \sum_e \sum_a P(B, j, m, e, a)$

which results in:

$$
P(b | j, m) = \alpha \sum_e \sum_a P(b) P(e) P(a | b, e) P(j | a) P(m | a)
$$
		

### Approaches to Uncertain Reasoning

* Default reasoning, which beliefs a conclusion until a better reason is found to believe something else.

* Rule-based approaches to uncertainty has also been tried.

* Fuzzy logic, which proposes an ontology that allows vagueness: that a proposition can be "sort of" true.


#### Rule-based methods for uncertain reasoning

Three desirable properties:

* *Locality*: $A \implies B$ means that we can conclude B given evidence A, without concerning with other rules or other evidence.
* *Detachment:* Once a logical proof is found for a prop. B, the prop. can be used regardless of how it was derived. It can be detached from its justification. 
* *Truth-functionality:* The truth of complex sentences can be computed from the truth of the components. 

A truth-functionality systems acts as it also believes $Sprinkler \implies Rain$ if it has the two rules:

* $Sprinkler \implies WetGrass$
* $WetGrass \implies Rain$


#### Ignorance - Dempster-Shafer theory

Belief function, written $Bel(X)$.
Deals with the distinction between uncertainty and ignorance.

**Coin toss with magicians coin:**


Because of no evidence whether or not it will become heads or tails the belief of result would be:

* $Bel(Heads) = 0$
* $Bel(\not Heads) = 0$


If a expert tells the system that it is 90% sure that the coin is fair, then Dempster-Shafer theory gives $Bel(Heads) = 0.90 * 0.50 = 0.45$

#### Fuzzy set theory

Is a means of specifying how well an object satisfies a vague description, such as "Is Henrik tall?". Is it true if he is "190 cm"?

Most people would say True, and some might say False, and some sort of. 

**Fuzzy set theory is not a method foruncertain reasoning at all.** Rather, fuzzy set theory treats Tall as a fuzzy predicate and saysthat the truth value of Tall (Nate) is a number between 0 and 1, rather than being just trueor false.

Fuzzy logic is a method for reasoning with logical expressions describing membershipin fuzzy sets. For example, the complex sentence $Tall (Nate) \land Heavy(Nate)$ has a fuzzytruth value that is a function of the truth values of its components.

This is the standard rules:

* $T(A \land B) = min(T(A), T(B))$* $T(A \lor B) = max(T(A), T(B))$* $T(¬A) = 1 − T(A)$


*Fuzzy control* is a methodology for constructing control systems in which the mapping between real-valued input and output parameters is represented by fuzzy rules.

It provides a simple real-valued function to smoothly interpolate values. We can find out:

$P( Henrik\ is\ tall\ | Height=190)$.



## Chapter 15 - Probabilistic Reasoning over time

Agents in partially observable environments must be able to keep track of the current states, to the extent that their sensors allow. 

A changing world is modeled using a variable for each aspect of the world state at each point in time. The transition and sensor models may be uncertain: the transition model describes the probability distribution of the variables at time *t*, given the state of the world at past time, while sensors model describes the probability of each percept at time *t*, given the current state of the world. The following seciton will describe general structure of interference algorithms for temporal models. Then we describe three specific kinds of models: **hidden Markov models, Kalman filters**, and **dynamic Bayesian networks**. 


#### States and observations

* See world as series of snapshots, or **time slices**, which each contain a set of random variables, some observable and some not. 

We will:

* denote $X_t$ as the set of state variables at time *t*, which are assumed to be unobservable.
* denote $E_t$ as the set of state evidence variables that are observable 

**Umbrella problem**

You are the security guard stationed at a secret underground installation. You want to know whether it’s raining today, but your only access to the outside world occurs each morning when you see the director coming in with, or without, an umbrella.

For each day *t*, the set $E_t$ thus contains a single evidence variable $Umbrella_t$ or $U_t$ for short (whether the umbrella appears), and the set $X_t$ contains a single state variable $Rain_t$ or $R_t$ for short (whether it is raining).

We denote $U_{1:3}$ as the set of variables $U_1, U_2, U_3$.


#### Transition and sensor models

The transition model specifies the probability distribution over the lates state variables given the previous values, that is, $P(X_t | X_{0:t-1})$. But $X_{0:t-1}$ becomes big as *t* increases.

We solve this by making a **Markov assumption** - that the current state only depends on a *finite fixed number* of previous states.

The simplest is the **first-order Markov process/chain**, which every state depends on only the previous state. With this assumption we get the transition model:

$P(X_t | X_{0:t-1}) = P(X_t | X_{t-1})$

For the second-order Markov process we get the trans. model $P(X_t | X_{0:t-1}) = P(X_t | X_{t-2}, X_{t-1})$

We also need to assume a stationary process - that the conditional probability (trans. model) stays the same throughout time.


For the **sensor model / observation model**. The evidence variable $E_t$ could depend on previous variables, but we make a sensor Markov assumption:

$$
P(E_t | X_{0:t}, E_{0:t-1}) = P(E_t | X_t)
$$


#### Inference in Temporal Models

* **Filtering**:
	* Computing the **belief state**, given all evidence to date. Also called state estimation.

* **Prediction**:
	* The task of computing distribution over the *future* state, given all evidence to date. That is we wish to compute $P(X_{t+k} | e_{1:t})$ for some *k* > 0.
	
* **Smoothing**:
	* The task of computing distribution over a *past* state, given all evidence up to the present. That is, we wish to compute $P(X_k | e_{1:t})$ for some *k* such that 0 ≤ *k* ≤ *t*. 
	
* **Most likely explanation**:
	* Given a sequence of observations, we might wish to find the sequence of states that is most likely to have generated those observations. That is, we wish to compute $argmax_{x_{1:t}} P(x_{1:t} | e_{1:t})$
	* For example, if the umbrella is present the first three days, but not the fourth, the most likely explanation is that it rained the first three days, but not the fourth.

	
* **Learning**:
	* The transition and sensor model can be learned from observations. Inference provides an estimate of what transitions actually occurred and of what states generated the sensor reading and these estimates can be used to update the model. 

	
> Note that learning requires smoothing, rather than filtering. 


#### Filtering and prediction

We need to compute the result for $t + 1$ from the new evidence $e_{t+1}$,

$$
P(X_{t+1} | e_{1:t+1}) = f(e_{t+1}, P(X_t | e_{1:t}))
$$,

for some function f. This process is called **recursive estimation**.

We rearrange the formula: 

$$
P(X_{t+1} | e_{1:t+1}) = P(X_{t+1} | e_{1:t}, e_{t+1}) (split\ evidence)
$$

$$
= \alpha P(e_{t+1} | X_{t+1}, e_{1:t}) P(X_{t+1} | e_{1:t}) (Bayes'\ rule)
$$

$$
= \alpha P(e_{t+1} | X_{t+1}) P(X_{t+1} | e_{1:t}) (sensor\ Markov\ assumption)
$$

> First term is obtainable from sensor model. Second term is a prediction of the next state. We obtain the next-step prediction by conditioning on the current state $X_t$:

$$
P(X_{t+1} | e_{1:t+1}) = \alpha P(e_{t+1} |X_{t+1}) \sum_{x_t} P(X_{t+1} | x_t, e_{1:t}) P(x_t | e{1:t})
$$

$$
= \alpha P(e_{t+1} |X_{t+1}) \sum_{x_t} P(X_{t+1} | x_t) P(x_t | e{1:t}) (Markov \ assumption)
$$

We look at the filtered estimate $P(X_t | e_{1:t})$ as a "message" $f_{1:t}$ that is propagated forward along the sequence, modeified by each transition and updated by each new observation. The process is given by,

$$
f_{1:t+1} = \alpha \ FORWARD(f_{1:t}, e_{1:t})
$$

where $FORWARD$ implements the update described above. 

The process begins with $f_{1:0} = P(X_0)$.

Prediction can be seen simply as filtering without the addition of new evidence. In fact, the filtering process already incorporates a one-step prediction, and it is easy to derive the following recursive computation for predicting the state $t + k + 1$ from a prediction for $t + k$:

$$
P(X_{t+k+1} | e_{1:t}) = \sum_{x_{t+k}} P(X_{t+k+1} | x_{t+k})P(x_{t+k} | e_{1:t})
$$


We use forward recursion to compute the **likelihood** of the evidence sequence, $P(e_{1:t})$. We use the likelihood message $L{1:t}(X_t) = P(X_t, e_{1:t})$.


We use: $l_{1:t+1} = FORWARD(l_{1:t}, e_{t+1})$. And with this sum the actual likelihood.


#### Smoothing

Recalculating $P(X_k | e_{1:t})$ for $0 ≤ k ≤ t$.

$$
P(X_k | e_{1:t}) = P(X_k | e_{1:k}, e_{k+1:t})
$$
$$= \alpha P(X_k | e_{1:k}) P(e_{k+1:t} |X_k, e{1:k}) (using \ Bayes’ \ rule)= \alpha P(X_k | e_{1:k}) P(e_{k+1:t} |X_k) (using \ conditional \ independence)
$$
$$= α f_{1:k} × b_{k+1:t}
$$

,where x is pointwise multiplication of vectors.

The backward message is defined as,

$$
b_{k+1:t} = P(e_{k+1:t} | X_k)
$$

$$
= \sum_{x_{k+1}} P(e_{k+1} | x_{k+1}) P(e_{k+2:t} | x_{k+1}) P(x_{k+1} | X_k)
$$

It is initialised with b_{t+1:t} = 1


This together is called the **forward-backward algorithm** which filters forwards from 0 to t and smoothes estimates backwards from t to 0.


> Often uses fixed-lag smoothing, which only calculates smoothed estimates for a fixed number *d*, e.g. slice by d steps behind the current time *t*. 

#### Most likely sequence

Suppose that [true, true, false, true, true] is the umbrella sequence. What is the weather sequence most likely to explain this? 


We get the relationship:

$$
max_{x_1...x_t} P(x_1,...,x_t, X_t+1 | e_{1:t+1})
$$

$$
= \alpha P(e_{t+1} | X_{t+1}) max_{x_t} 
P(X_{t+1} | x_t) max_{x_1...x_t} P(x_1,...,x_{t-1}, x_t | e_{1:t})
$$

Here the forward message is:

$$
m_{1:t} = max_{x_1...x_t} P(x_1,...,x_{t-1}, X_t | e_{1:t})
$$

 
 
## Chapter 16 - Making Simple Decisions



Decision theory in its simplest form, deals with choosing among actions based on the desirability of their *immediate* outcomes; that is, the environment is assumed to be episodic. 

The agent	s preferences are captured by a **utility function**, $U(s)$, which assigns a single number to express the desirability of a state. The **expected utility** of an action given the evidence, $EU( a | e)$ where a is a action and e is a set of evidence observations, is just the average utility value of the outcomes, weighted by the probability that the outcome occurs:

$$
EU(a | e) = \sum_{s'} P(RESULT(a) = s' | a, e) U(s')
$$

The principle of **maximum expected utilty** (MEU) says that a rational agent should choose the action that maximizes the agent's expected utility; $action = argmax_{a} EU(a | e)$


#### Utility theory

We use the following notation to describe an agent’s preferences:
* $A \succ B$ the agent prefers A over B.* A ∼ B the agent is indifferent between A and B.* $A \succsim B$ the agent prefers A over B or is indifferent between them.



##### Lottery example

A lottery $L$ with possible outcomes $S_1,.., S_n$ that occur with probabilities $p_1,..., p_n$ is written

$$
L = [p_1, S_1; p_2, S_2;...; p_n, S_n]
$$


List of contraint that we require any reasonable preference relation to obey:

* **Orderability:**
	* Given two lotteries, a rational agent must either prefer one to the other or else rate the two as equally preferable.	

* **Transitvity:**
	* Given three lotteries, if an agent prefers A to B, and prefers B to C, then the agent must prefer A to C.
	
* **Continuity:**
	* If some lottery B is between A and X in preference, then there is some probability *p* for which the rational agent will be indifferent between getting B for sure and the lottery that yields A with probability *p* and C with probability 1 - *p*
	
* **Substitutability:**
	* If an agent is indifferent between two lotteries A and B, then the agent is indifferent between two more complex lotteries that are the same except that. If you switch each of them out separately and compare with another lottery, the outcome would be the same for both lotteries. 

* **Monotonicity:**
	* Suppose two lotteries have the same two possible outcomes, A and B. If an agent prefers A to B, then the agent must prefer the lottery that has a higher probability for A (and vice versa).

* **Decomposability:**
	* Compound lotteries can be reduced to simpler ones using the laws of probability. This has been called the "no fun in gambling" rule because it says that two consecutive lotteries can be compressed into a single equivalent lottery. 

	
#### Preferences lead to utility

*Axioms of utility:*

* Existence of Utility Function:
	* $U(A) > U(B) \Leftrightarrow A \succ B$
	* $U(A) = U(B) \Leftrightarrow A ∼ B$

* Expected utility of a Lottery:
	* $U ([p_1, S_1;...; p_n, S_n]) = \sum_i p_i U(S_i)$

As in game-playing, in a deterministic environment an agent just needs a preference ranking on states. This is called a **value function**.



#### Utility Functions

Utility is a function that maps from lotteries to real numbers. Utility functions represents preferences. 


For a utility function, a scale is needed. You can provide a utility for the "best possible prize", and a "worst possible catastrophe".

Normalized utilities use a scale with worst at 0 and best at 1. 

To assess a utility for any prize $S$, you could ask the agent to choose between the prize $S$ and a standard lottery $[p, "worst"; (1-p), "best"]$, and change $p$ until the agent is indifferent with S and the standard lottery. Assuming normalized utilities we have that $U(S) = p$.


* Micromort = 1/1 million chance of death
* QALY - quality-adjusted life year. Patients with a disability are willing to accept a shorter life expectancy to be restored to full health. 

**Expected monetary value (EMV)**:

Assuming winning a contest, either getting 1 million dollar, or coin toss for either losing all money or winning a total of 2.5 million dollars.

The EMV is,
$$
\frac{1}{2}(\$ 0) + \frac{1}{2}(\$ 2,500,000) = \$ 1,250,000
$$
$

Which choice is not easy, but could be solved by looking at expected utilities ($S_k$ is your current wealth):

* $EU(Accept) = \frac{1}{2}U(S_k) + \frac{1}{2}U(S_{k+2,500,000}$
* $EU(Decline) = U(S_{k+1,000,000}$

Since there is a risk in not getting the money, we have that
* $U(L) < U(S_{EMC(L)})$

The utility function for monetary prizes looks like a horizontal S. The region at large negative wealth has a *risk-seeking* behaviour. 

#### Expected utility

The rational way to choose the best action, $a^*$, is to maximize expected utility:

$$
a^* = argmax_{a} EU(a | \textbf{e})
$$

Decision theory is a normative theory: it describes how a rational agent *should* act.

A descriptive theory, on the other hand, describes how actual agents really do act. 


**Certainty effect**: People are strongly attracted to gains that are certain. There are several reasons why this may be so. First, people know they will experience *regret* if they gave up certain money. Second, it is less computational heavy to choose the sure thing.

People might give up EMV-value to be sure that they will get a prize. 

Most people elect the known probability rather than the unknown unknowns. 


**Framing effect:** "90% survival rate" over "10% death rate" - people like the first one twice as much. 


**Anchoring effect:** Make it seem like your choices seem good relatively compared to other. Example, a restaurant offering a wine bottle of $\$$ 200, that it knows nobody will buy, but which makes a bottle of $\$$50 wine seem like a bargain.$


#### Multi-attribute Utility Functions

Siting a new airport requires consideration of the disruption caused by construction; the cost of land; the distance from centers of population; the noise of flight operations; safety issues and so on. Problems like these, in which outcomes are characterized by two or more attributes, are handled by **multi-attribute utility theory**.

We call the attributes $\textbf{X} = X_1,..., X_n$; a complete vector of assignments will be $\textbf{x} = 〈x_1,.., x_n〉$, where each $x_i$ is either a numeric value or a discrete value with an assumed ordering on values. 


**Dominance**:

Suppose that airport site $S_1$ costs less, generates less noise pollution, and is safer than site $S_2$. One would not hesitate to reject $S_2$. We then say that there is strict dominance of $S_1$ over $S_2$.

* If $A_1$ stochastically dominates $A_2$, then for any monotonically nondecreasing utility function $U(x)$, the expected utility of $A_1$ is at least as high as the expected utility of $A_2$.

> Cumulative distribution measures the probability that the cost is less than or equal to the given amount - that is, it integrates the original distribution.


### The multiattrbute utility function

Suppose we have *n* attributes, each of which has *d* distinct possible values. TO specify the complete utility function $U(x_1,..., x_n)$, we need $d^n$ values in the worst case.

* $U(x_1,..., x_n) = F[f_1(x_1),..., f_n(x_n)]$
	* where $F$ is, we hope, a simple function such as addition. 



#### Preferences without uncertainty
	
The basic regularity that arises in deterministic preference structures is called preference independence. Two attributes $X_1$ and $X_2$ are preferentially independent of a third attribute $X_3$ if the preference between outcomes 〈$x_1$, $x_2$, $x_3$〉 and 〈$x_1'$, $x_2'$, $x_3$〉, does not depend on the particular value $x_3$ for attribute $X_3$.


We say that the set of attributes ${Noise,\ Cost\ ,\ Deaths}$ exhibits mutual preferential independence (MPI). MPI says that, whereas each attribute may be important, it does not affect the way in which one trades off the other attributes against each other.

If attributes $X_1,..., X_n$ are mutually preferentially independent, then the agent's preference behaviour can be described as maximizing the function:

$$
V(x_1,...,\ x_n) = \sum_i V_i(x_i)
$$

, where each $V_i$ is a value function referring only to the attribute $X_i$.

Example:

$V(noise,\ cost,\ deaths) = -noise × 10^4 - cost - deaths × 10^{12}$

A value function of this type is called an **additive value function**. 


#### Preferences with uncertainty


The basic notion of **utility independence** extends preference independence to cover lotteries: a set of attributes $\textbf{X}$ is utility independent of a set of attributes $\textbf{Y}$ if preferences between lotteries on the attributes in $X$ are independent of the particular values of attributes in $Y$.


A set of attributes is mutually utility independent (MUI) if each of its subsets is utility-independent of the remaining attributes.

Dvs. velg mellom $X$ og $Y$, vil gi samme utfall uavhengig av verdien til $Z$.

MUI implies that the agent's behaviour can be described using a **multiplicative utility function**.

The general form of a multiplicative utility function is best seen by looking at the case for three attributes. For conciseness, we use Ui to mean Ui(xi):

$$U = k_1 U_1 + k_2 U_2 + k_3 U_3 + k_1 k_2 U_1 U_2 + k_2 k_3 U_2 U_3 + k_3 k_1 U_3 U_1
$$
$$+ k_1 k_2 k_3 U_1 U_2 U_3
$$


### Decision Networks

* **Influence diagram** $\implies$ **decision network**.

Representing a decision problem with a decision network:

* **Change nodes (ovals):**
	* Represent random variables. The agent could be uncertain about the construction cost.

* **Decision nodes (rectangles):**
	* Represent points where the decision maker has a choice of action. In this case, the $AirportSite$ action can take on a different value for each site under consideration. The choice influences the cost, safety and noise that will result. 

* **Utility nodes (diamonds):**
	* Represent the agent's utility function. the utility node has as parents all variables describing the outcome that directly affect utility. It needs a description of the agents utility function, such as a tableau, additive or linear function of the attribute values.

	

### Value of Information

one of the most important parts of decision making is knowing what questions to ask. 

#### Oil example

Suppose an oil company is hoping to buy one of *n* indistinguishable blocks of ocean-drilling rights. Let us assume further that exactly one of the blocks contains oil worth C dollars, while the others are worthless. The asking price of each block is $C/n$ dollars.


Now suppose that a seismologist offers the company the results of a survey of blocknumber 3, which indicates definitively whether the block contains oil. How much shouldthe company be willing to pay for the information? The way to answer this question is to examine what the company would do if it had the information:

* With probability $1/n$, the survey will indicate oil in block 3. In this case, the company will buy block 3 for $C/n$ dollars and make a profit of $C$ − $C/n$ = $(n − 1)C/n$ dollars.
* With probability $(n−1)/n$, the survey will show that the block contains no oil, in whichcase the company will buy a different block. Now the probability of finding oil in oneof the other blocks changes from $1/n$ to $1/(n−1)$, so the company makes an expectedprofit of $C/(n − 1) − C/n = C/n(n − 1)$ dollars.

Expected profit, given the survey information:

$$
\frac{1}{n} × \frac{(n-1)X}{n} + \frac{n-1}{n}× \frac{C}{n(n-1)} = C/n
$$

The information is worth at most the value of the block itself.

#### Value of Perfect Information (VPI)

Let the agent's initial evidence be $\textbf{e}$. Then the value of the current best action $\alpha$ is defined by

$$
EU(\alpha | \textbf{e}) = max_a \sum_{s'} P(RESULT(a) = s' | a, \textbf{e}) U(s')
$$

and the value of the new best action (after the new evidence $E_j = e_j$ is obtained) will be

$$
EU(\alpha_{e_j} | \textbf{e}, e_j) = max_a \sum_{s'} P(RESULT(a) = s' | a, \textbf{e}, e_j) U(s')
$$

But $E_j$ is a random variable whose value is currently unknown, so to determine the value of discovering $E_j$ , given current information e we must average over all possible values $e_jk$ that we might discover for $E_j$ , using our current beliefs about its value:

$$
VPI_e(E_j) = \Big{(} \sum_k P(E_j = e_{jk} | \textbf{e}) EU(\alpha_{e_{jk}} | \textbf{e}, E_j=e_{jk}) \Big{)} - EU(\alpha | \textbf{e})
$$


In sum, information has value to the extent that it is likely to cause a change of planand to the extent that the new plan will be significantly better than the old plan.


> The expected value of information is nonnegative!
> 
> It is also not additive:
> $VPI_e(E_j,E_k) \neq VPI_e(E_j) + VPI_e(E_k)$


### Decision-Theoretic Systems

A decision-theoretic expert system for this problem can be created with the following process. The process can be broken down into the following steps:

* Create a causal model
* Simplify to a qualitative decision model
* Assign probabilities
* Assign utilites
* Verify and refine model
	* To evaluate system we need a set of correct (input, output) pairs; a so-called *gold standard* to compare against. 



**Summary:**

* Probability theory + Utility theory = Decision theory



## Chapter 17 - Making Complex Decisions

Whereas Chapter 16 was concerned with one-shot or episodic decision problems, in which the utility of each action’s outcome was well known, we are concerned here with sequential decision problems, in which the agent’s utility depends on a sequence of decisions.

### Sequential Decision Problems

Environment:

| | | | +1 |
|---|---|---|---|
| | | | -1 |
| | Wall | | |
| Start | | | |

Actions:
* 0.8 of going in intended direction
* 0.1 in going in either left or right to the intended direction

Possible solution: $[Up, Up, Right, Right, Right]$, goes up and around the barrier and into the goal with a probability of $0.8^5 = 0.32768$. There is also a small chance that it will go the other way around, $0.1^4 \times 0.8 = 0.32776$.


* Outcome is stochastic $\implies$ Dynamic Bayesian network


We need a utility function, and because the decision problem is sequential, the utility function will depend on a sequence of states - an environmental history - rather than on a single state.


In each state *s*, the agent receives a **reward** $R(s)$, which may be positive or negative, but must be bounded. 

> For this example, the reward is $-0.04$ in all states except the terminal states (which have rewards $+1$ and $-1$). 


The utility of an environment history is just (for now) the sum of the rewards received. 

**Summed up:**

A sequential decision problem for a fully observable, stochastic environment with a Markovian transition model and additive rewards is called a **Markov decision process** or **MDP**, and consists of a set of states (with initial stat $s_0$); a set $ACTIONS(s)$ of actions in each state; a transition model $P(s' | s, a)$; and a reward function $R(s)$.


Since it is a stochastic environment, the agent could get in any state. Therefore, a solution must specify what the agent should do for *an* state that the agent might reach. This is called a **policy**, denoted $/pi$, and $\pi(s)$ is the action recommended by the policy $\pi$ for state $s$.

> An optimal policy is a policy that yields the highest expected utility.


##### Utilities over time

* Finite horizon: There is a *fixed* time N after which nothing matters - the game is over, so to speak. Thus, $U_h([s_0, s_1,..., s_{N+k}]) = U_h([s_0, s_1,..., s_N])$for all $k > 0$.

	* Suppose that $N = 3$, then to have any chance of reaching the $+1$ state, the agent must head directly for it, and the optimal action is to go $Up$. 
	* On the other hand, if $N = 100$, then there is plenty of time to take the safe route by going $Left$. 

*So with a finite horizon, the optimal action in a given state could change over time.*


We say that the optimal policyfor a finite horizon is **nonstationary**. With no fixed time limit, on the other hand, there is no reason to behave differently in the same state at different times - thus **stationary** policy, which is simpler.


Stationarity is a fairly innocuous-looking assumption with very strong consequences: it turns out that under stationarity there are just two coherent ways to assign utilities to sequences:

1. **Additive rewards:**
	* 	The utility of a state sequence is $U_h([s_0, s_1,...]) = R(s_0) + R(s_1) + ···$.

	
2. **Discounted rewards:**
	* The utility of a state sequence is $U_h([s_0, s_1, s_2,...]) = R(s_0) + γ R(s_1) + γ^2 R(s_2) + ···$
	* Where the discount factor γ is a number between 0 and 1. The discount factor describesthe preference of an agent for current rewards over future rewards.


The expected utility, with discounted rewards obtained by executing $\pi$ starting in $s$ is given by

$$
U^{\pi}(s) = E \Big{[} \sum_{t=0}^{\inf} γ^t R(S_t) \Big{]}
$$

### Value Iteration

Value iteration is an algorithm used for calculating an optimal policy. The basic idea is to calculate the utility of each state, and then use the state utilities to select an optimal action in each state.

the utility of a state is the immediate reward for that state plus the expected discounted utility of the next state, assuming that the agent chooses the optimal action. That is, the utility of a state is given by

$$
U(s) = R(s) + γ max_{a \in A(s)} \sum_{s'} P(s' | s, a) U(s')
$$


Example with 3x4-world:

$$
U(1, 1) = −0.04 + γ max[
$$

$$
0.8U(1, 2) + 0.1U(2, 1) + 0.1U(1, 1), \ (Up)
$$

$$0.9U(1, 1) + 0.1U(1, 2),\ (Left )
$$

$$0.9U(1, 1) + 0.1U(2, 1),\ (Down)
$$

$$0.8U(2, 1) + 0.1U(1, 2) + 0.1U(1, 1) ]\ (Right )
$$.


We start with arbitrary initial values for the utilities, calculate the right-hand side of the equation, and plug it into the left-hand side—thereby updating the utility of each state from the utilities of its neighbors. We repeat this until we reach an equilibrium

**Convergence**:

The basic concept used in showing that value iteration converges is the notion of a **contraction**. Roughly speaking, a contraction is a function of one argument that, when applied to two different inputs in turn, produces two output values that are "closer together". 

> Example: Divide by two is a contraction, and will in the end converge to zero. (A fixed point).


It measures the "length" of a vector by the absolute value of its biggest component:

$$
||U|| = max_s | U(s) |
$$

The distance between two vectors, $|| U - U' ||$ is the maximum difference between any two corresponding elements.

That is, the Bellman update is a contraction by a factor of $γ$ on the space of utility vectors.


### Policy Iteration

The policy iteration algorithm alternates the following two steps, beginning from some initial policy $\pi_0$:

* **Policy evaluation:** given a policy $\pi_i$, calculate $U_i = U^{\pi_i}$, the utility of each state if $\pi_i$ were to be executed.
* **Policy improvement:** Calculate a new MEU (Maximum Expected Utility) policy $\pi_{i+1}$, using one-step look-ahead based on $U_i$.

The algorithm terminates when the policy improvement step yields no change in the utilities.

At this point, we know that the utility function Ui is a fixed point of the Bellman update, so it is a solution to the Bellman equations, and πi must be an optimal policy.

It is possible to, on each iteration, choose any subset og states and apply either kind of updating to that subset. Because there are som states that is not worth calculating (no need to know the exact utility of jumping off a cliff).


#### Partially observable multivariable decision problems

The real world is a POMDP - partially observable multivariable decision problem. 

A POMDP has the sameelements as an MDP—the transition model $P(s' | s, a)$, actions $A(s)$, and reward function $R(s)$ — but, like the partially observable search problems, it also has a sensor model $P(e | s)$. 

Here, as in Chapter 15, the sensor model specifies the probability of perceivingevidence $e$ in state $s$. 

In POMDPs, the belief state b becomes aprobability distribution over all possible states

belief state for the 4×3 POMDP could be the uniform distribution over the nine nonterminalstates, i.e., $〈 \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, \frac{1}{9}, 0, 0〉$

We dwrite $b(s)$ for the probability assigned to the actual state $s$ by belief state $b$. 

The agent can calculate its current belief state as the conditionalprobability distribution over the actual states given the sequence of percepts and actions so far. This is essentially the filtering task described in Chapter 15.

For POMDPs, we also have an action to consider, but the result is essentially the same. If $b(s)$ was the previous belief state, and the agent does actiona and then perceives evidence e, then the new belief state is given by


$$
b'(s') = \alpha P(e | s') \sum_s P(s' | a, a) b(s)
$$

By analogy with the update operator for filtering, we can write this as 

$$
b' = FORWARD(b, a, e)
$$

The fundamental insight required to understand POMDPs is this: *the optimal action depends only on the agent's current belief state.*

Hence, the decision cycle of a POMDP agent can be broken down into thefollowing three steps:
1. Given the current belief state b, execute the action $a = \pi^*(b)$.2. Receive percept $e$.3. Set the current belief state to $FORWARD(b, a, e)$ and repeat.

Not a policy, but a **conditional** plan, since it is partially observable and perhaps also nondeterministic.

For general POMDPs, however, finding optimal policies is very difficult (PSPACE hard, in fact — i.e., very hard indeed).


## Chapter 18 - Learning from Examples

An agent is **learning** if it improves its performance on future tasks after making observations about the world. 

#### Forms of learning

Any component of an agent can be improved by learning from data. The improvements depend on four major factors:

* Which *component* is to be improved
* What *prior knowledge* the agent already has.
* What *representation* is used for the data and the component.
* What *feedback* is available to learn from.


Example, a self-driving car agent can learn when to brake the car, by feedback from an instructor.


**Representation and prior knowledge**:

We have seen several examples of representations for agent components: propositional andfirst-order logical sentences for the components in a logical agent; Bayesian networks for the inferential components of a decision-theoretic agent, and so on. Effective learning algorithms have been devised for all of these representations. This chapter (and most of current machine learning research) covers inputs that form a factored representation—a vector of attribute values—and outputs that can be either a continuous numerical value or a discrete value.

**Feedback to learn from:**

There are three *types of feedback* that determine three main types of learning:

* **Unsupervised learning**: the agent learns patterns in the input even though no explicit feedback is supplied. (Most common task is clustering - detecting potentially useful clusters of input example).

* **Reinforcement learning**: the agent learns from a series of reinforcements - rewards or punishments. For example the lack of tip from a taxi customer. 

* **Supervised learning**: the agent observes some input-output pairs and learns a function that maps from input to output.


### Supervised Learning

*The task of supervised learning is this:*

Given a **training set** of $N$ example input-output pairs $(x_1, y_1),(x_2, y_2),..(x_n, y_n)$, where each $y_i$ was generated by an unknown function $y = f(x)$, discover a function $h$ that approximates the true function $f$.


The function $h$ is a **hypothesis**. Learning is a search though the space of possible hypotheses for one that will perform well, even on new examples beyond the training set.

To measure the accuracy of a hypothesis we give it a **test set** of examples that are distinct from the training set. 


When the output $y$ is one of a finite set of values (such as *sunny*, *cloudy* or *rainy*), the learning problem is called **classification**, and is called Boolean or binary classification if there are only two values. When y is a number (such as tomorrow’s temperature), the learning problem is called **regression**.



*In general, there is a tradeoff between complex hypotheses that fit the training data well and simpler hypotheses that may generalize better.*

We say that a learning problem is realizable if the hypothesis space contains the true function.



> There is a tradeoff between the
 expressiveness of a hypothesis spaceand the complexity of finding a good hypothesis within that space. For example, fitting a straight line to data is an easy computation; fitting high degree polynomials is somewhat harder; and fitting Turing machines is in general undecidable.


### Learning Decision Trees

Decision tree induction is one of the simplest and yet most successful forms of machine learning.

A **decision tree** represents a function that takes as input a vector of attribute values and returns a "decision" - a single output value. The input and output values can be discrete or continuous.

For now, Boolean classification, where each example of input is either a true or negative example. 

A decision tree reaches its decision by performing a sequence of tests. Each internal node in the tree corresponds to a test of the value of one of the input attributes, $A_i$, and the branches from the node are labeled with the possible values of the attribute, $A_i = v_{ik}$.

Each leaf node in the tree specifies a value to be returned by the function. 

As an example, we will build a decision tree to decide whether to wait for a table at a restaurant. The aim here is to learn a definition for the goal predicate $WillWait$. 

First we list the attributes that we will consider as part of the input:



1. $Alternate$: whether there is a suitable alternative restaurant nearby.2. $Bar$ : whether the restaurant has a comfortable bar area to wait in.3. $Fri/Sat$: true on Fridays and Saturdays.4. $Hungry$: whether we are hungry.5. $Patrons$: how many people are in the restaurant (values are None, Some, and Full ).6. $Price$: the restaurant’s price range ($\$, \$\$, \$\$\$ $).7. $Raining$: whether it is raining outside.8. $Reservation$: whether we made a reservation.9. $Type$: the kind of restaurant (French, Italian, Thai, or burger).10. $WaitEstimate$: the wait estimated by the host (0–10 minutes, 10–30, 30–60, or >60).


A Boolean decision tree is logically equivalent to the assertion that the goal attribute is true if and only if the input attributes satisfy one of the paths leading to a leaf with value true.

We want a tree that is consistent with the examples and is as small as possible.

The $DECISION-TREE-LEARNING$ algorithm adopts a greedy divide-and-conquerstrategy: always test the most important attribute first. This test divides the problem up into smaller subproblems that can then be solved recursively. By “most important attribute,” we mean the one that makes the most difference to the classification of an example.

```python
function DECISION-TREE-LEARNING(examples, attributes, parent_examples):
    if examples is empty return PLURALITY-VALUE(parent examples):
    else if all examples have the same classification return the classification:
    else if attributes is empty then return PLURALITY-VALUE(examples):
    else:
        A ← argmax(a in attributes)IMPORTANCE(a, examples)
        tree ← a new decision tree with root test A
        for each value vk of A do:
            exs ← {e: e in examples and e.A = vk}
            subtree ← DECISION-TREE-LEARNING(exs, attributes - A, examples)
            add a branch to tree with label (A = vk) and subtree subtree
        return tree

```

We can evaluate the accuracy of a learning algorithm with a **learning curve**.

The greedy search used in decision tree learning is designed to approximately minimize the depth of the final tree. The idea is to pick the attribute that goes as far as possible toward providing an exact classification of the examples.

We will use the notion of informationgain, which is defined in terms ENTROPY of **entropy**.

*Entropy* is a measure of the uncertainty of a random variable; acquisition of information corresponds to a reduction in entropy. 

> A flip of a fair coin is equally likely to come up heads or tails, 0 or 1, and we will soon show that this counts as “1 bit” of entropy. 
> The roll of a fair four-sided die has 2 bits of entropy, because it takes two bits to describe one of four equally probable choices.


We define $B(q)$ as the entropy of a Boolean random variable that is true with probability $q$:

$$
B(q) = -(qlog_2(q) + (1-q)log_2(1-q))
$$

The information gain from the attribute test on A is the expected reduction in entropy:

**We say that the information gain on an attribute split is the current entropy minus the mean sum of the new subsets of examples.**

Overfitting occurs with all types of learners, even when the target function is not at all random. We can see polynomial functions overfitting data if the polynomial function is more complex than the data itself, and less generalizing.

Overfitting becomes more likely as the hypothesis space and the number of input attributes grows, and less likely as we increase the number of training examples.

**Decision tree pruning** combats overfitting. Starting with a full tree, we look at a test node that has only leaf nodes as descendants. If the test appears to be irrelevant—detecting only noise in the data — then we eliminate the test, replacing it with a leaf node.

#### Evaluating and Choosing the best Hypothesis

We want to learn a hypothesis that fits the future data best. We make the stationarity assumption: that there is a probability distribution over examples that remains stationary over time.

Each example datapoint is a random variable $E_j$ whose observed value $e_j =(x_j, y_j)$ is sampled from that distribution, and is independent of the previous examples, each with the identical prior probability distribution. 


**Accuracy:**

To get an accurate evaluation of ahypothesis $h$, we need to test it on a set of examples it has not seen yet. The simplest approach is the one we have seen already: randomly split the available data into a training set from which the learning algorithm produces $h$ and a test set on which the accuracy of $h$ is evaluated.


We can use *k***-fold cross-validation** to get an accurate estimate and squeezing more out of the data. The idea is that each example serves double duty - as training data and test data. First we split the data into $k$ equal subsets. We then perform $k$ rounds of learning; on each round $1/k$ of the data is held out as a test set and the remaining examples are used as training data. 

*We see typical curves in overfitting:* the training set error decreases monotonically (although there may in general be slight random variation), while the validation set error decreases at first, and then increases when the model begins to overfit.


### From error rates to loss

In machine learning it is traditional to express utilities by means of a **loss function**. The loss function $L(x, y, \hat{y})$ is defined as the amount of utility lost by predicting $h(x) = \hat{y}$ when the correct answer is $f(x) = y$


* $L(spam, nospam) = 1$
* $L(nospam, spam) = 10$

Simple loss functions:

* *Absolute value loss:* $L_1(y, \hat{y}) = |y - \hat{y}|$
* *Squared value loss:* $L_2(y, \hat{y}) = (y - \hat{y})^2$
* *0/1 loss:* $L_{0/1}(y, \hat{y}) = 0 if y = \hat{y}, else 1$

**Empirical loss** on a set of examples, $E$:

$$
EmpLoss_{L, E}(h) = \frac{1}{N} \sum_{(x,y) \in E} L(y, h(x))
$$


The estimated best hypothesis \hat{h}^* is then the one with the minimum empirical loss.


> The process of explicitly penalizing complex hypotheses is called regularization


#### Feature selection

Another way to simplify models is to reduce the dimensions that the models work with.A process of **feature selection** can be performed to discard attributes that appear to be irrelevant.


### Artificial Neural Networks


![Neuron](https://i.imgur.com/92qgm4K.png)

Above is a simple mathematical model of a neuron. The unit's output activation is $a_j = g(\sum_{i=0}^n w_{i,j}a_i$.

A neural network is just a collection of units connected together; the properties of the network are determined by its topology and properties of the "neurons".


#### NN structures

Neural networks are composed of nodes or **units** connected by directed **links**. A link from unit *i* to unit *j* serves to propagate the **activation** $a_i$ from *i* to *j*. Each link also has a numeric **weight** $w_{i,j}$ associated with it, which determines the strength and sign of the connection.

Each unit *j* first computes a weighted sum of its input:

$$
in_j = \sum_{i=0}^n w_{i,j}a_i
$$

then it applies an **activation function** $g$ to this sum to derive the output:
 
$$
a_j = g(in_j) = g\Big{(} \sum_{i=0}^n w_{i,j}a_i \Big{)}
$$

A **feed-forward network** has connections only in one direction - that is, it forms a directed acyclic graph. 

Each node receives input from "upstream" nodes and delivers output to "downstream" nodes. A feed-forward network represents a function of its current input; this, it has no internal state other than the weights. 

A **recurrent network** feeds its outputs back into its own inputs, which means that the activation levels of the network form a dynamical system that may reach a stable state or exhibit oscillations or even chaotic behaviour.

Hence, recurrent networks can support short-term memory. This makes them more interesting as models of the brain, but also more difficult to understand. 


**Layers:**

Feed-forward networks are usually arranged in *layers*, such that each unit receives input only from units in the immediately preceding layer. **Hidden layers** are layers that are not connected to the outputs of the network. 


**Perceptron:**

A network with all the inputs connected directly to the outputs is called a **single-layer neural network**, or a **perceptron network**. 

Perceptron networks can only learn linearly separable functions, and cannot learn functions as XOR. 


#### Multilayer feed-forward neural networks

Having a network with multiple layers, we can find express the output as a function of the input and the weights.

As long as we can calculate the derivatives of such expressions with respect to the weights, we can use the **gradient-descent loss minimization** method to train the network.

And because the function represented by a network can by highly nonlinear - composed, as it is, of nesten nonlinear soft threshold function - we can see neural networks as a tool for doing *nonlinear regression*. 

#### Learning in multilayer networks

We should look at the network as implementing a vector function $textbf{h_w}$ rather than a scalar function $h_w$.

For learning, we need to update all weights, and those updates will depend on errors from their downstream nodes' errors. 

This dependency is very simple in the case of any loss function that is *additive* across the components of the error vector $\textbf{y} - \textbf{h_w(x)}$. For the $L_2$ loss, we have, for any weight $w$:

$$
\frac{\partial}{\partial w} Loss(w) = \frac{\partial}{\partial w} |y - h_w(x|^2 = \frac{\partial}{\partial w} \sum_k (y_k - a_k)^2
$$

Each term in the final summation is just the gradient of the loss for the *k*th output, computed as if the other outputs did not exist. 

We can then **back-propagate** the error from the output layer to the hidden layers The back-propogation process emerges directly from a derivation of the overall error gradient. 

Let $Err_k$ be the *k*th component of the error vector $y - h_w$. We define a modified error $\Delta_k = Err_k * g'(in_k)$, so that the weight-update rule becomes:

$$
w_{j,k} \leftarrow w_{j,k} + \alpha * a_j * \Delta_k
$$

For updating the weights between input-layer and first hidden layer we need to use another update-function, which I will not explain in detail.



If we want to consider networks that are not fully connected, then we need to find some effective search method through the very large space of possible connection topologies.The optimal brain damage algorithm begins with a fully connected network and removes connections from it. After the network is trained for the first time, an information-theoretic approach identifies an optimal selection of connections that can be dropped. The network is then retrained, and if its performance has not decreased then the process is repeated. In addition to removing connections, it is also possible to remove units that are not contributing much to the result.

### Nonparametic models

Linear regression and neural networks use the training data to estimate a fixed set of parameters $w$. That defines our hypothesis $h_w(x)$, and at that point we can throw away the training data, because they are all summarized by $w$.

A learning model that summarizes data with a set of parameters of fixed size (independent of the number of training examples) is called a **parametric model**.

A **nonparametric model** is one that cannot be characterized by a bounded set of parameters. For example, suppose that each hypothesis we generate simply retains within itself all of the training examples and uses all of them to predict the next example.


![K-nearest neighbour](https://i.imgur.com/KBY2bDK.png)


#### Nearest neighbor models

Given a query $x_q$, find the *k* examples that are *nearest* to $x_k$. This is called *k***-nearest neighbors** lookup. I'll use the notation $NN(k, \textbf{x_q})$ to denote the set of *k* nearest neighbors.

To do classification, first find $NN(k, \textbf{x_q})$, then take the plurality vote of the neighbors. To avoid ties, *k* is always chosen to be an odd number. 

To do regression, we can take the mean or median of the *k* neighbors.

Typically distances are measured with a **Minkowski distance** or $L^p$ norm, defined as 

$$
L^p (x_j, x_q) = \big{(} \sum_i | x_{j,i} - x_{q,i}|^p \big{)}^{1/p}
$$

With $p = 2$, this is Euclidean distance, and with $p = 1$ it is Manhattan distance. 


## Chapter 21 - Reinforcement Learning


The absence of feedback from a teacher, an agent can learn a transition model for its own moves and can perhaps learn to predict the opponent’s moves, but without some feedback about what is good and what is bad, the agent will have no grounds for deciding which move to make.

This kind of feedback is called a **reward**, or **reinforcement**. In games like chess, the reinforcement is received only at the end of the game. In other environments, the rewards come more frequently.

The task of **reinforcement learning** is to use observed rewards to learnan optimal (or nearly optimal) policy for the environment.



Reinforcement learning might be considered to encompass all of AI: an agent is placed in an environment and must learn to behave successfully therein.


### Passive Reinforcement Learning

The agent’s policy is fixed and the task is to learn the utilities of states (or state–action pairs); this could also involve learning a model of the environment.


In passive learning, the agent’s policy $\pi$ is fixed: in state $s$, it always executes the action $\pi(s)$. Its goal is simply to learn how good the policy is — that is, to learn the utility function $U_{\pi}(s)$.

The passive learning task is similar to the *policy evaluation task*, part of the *policy iteration* algorithm described for Chapter 17.

The main difference is that the passive learning agent *does not know the transition model* $P(s' | s, a)$, whichspecifies the probability of reaching state $s'$ from state $s$ after doing action $a$; nor does it know the reward function $R(s)$, which specifies the reward for each state.

![3x4-world](https://i.imgur.com/RDtQNNP.png)

The agent executes a set of **trials** in the environment using its policy $\pi$. In each trial, the agent starts in state (1,1), and experiences a sequence of state transitions until it reaches one of the terminal states. 

Its percepts supply both the *current state* and the *reward* received in that state.


##### Adaptive dynamic programming - ADP

An ADP agent takes advantage of the constraints among the utilities of states by learning the transition model that connects them and solving the corresponding Markov decision process using a dynamic programming method.

The first approach, **Bayesian reinforcement learning**, assumes a prior probability $P(h)$ for each hypothesis $h$ about what the true model is; the posterior probability $P(h | e) $ is obtained in the usual way by Bayes’ rule given the observations to date

Learning from transitions can be done with **temporal-difference**. When a transition occurs from state $s$ to state $s'$, we apply the following update to $U^{\pi}(s)$:

$$
U^{\pi}(s) \leftarrow U^{\pi}(s) + \alpha (R(s) + γU^{\pi}(s') - U^{\pi}(s))
$$

, $\alpha$ is the **learning rate** parameter.










### Active Reinforcement Learning


The agent must also learn what to do. The principal issue is **exploration**.

An active agent must decide what actions to take. Let us begin with the adaptive dynamic programming agent and consider how it must be modified to handle this new freedom.

Using **Bellman equations** described earlier to obtain the utility function $U$. 

#### Exploration

Greedy agent is an agent that sticks to a good policy, never learning the utilities of other states.

What the greedy agent has overlooked is that actions do more than provide rewards according to the current learned model; they also contribute to learning the true model by affecting the percepts that are received. 

By improving the model, the agent will receive greater rewards in the future. An agent therefore must make a tradeoff between exploitation to maximize its reward. With greater understanding, less exploration is necessary.

Any reasonable scheme needs to be greedy in the limit of infinite exploration, or **GLIE**. A GLIE scheme must try each action in each state an unbounded number of times to avoid having a finite probability that an optimal action is missed because of an unusually bad series of outcomes.

> There are several GLIE schemes; one of the simplest is to have the agent choose a random action a fraction $1/t$ of the time and to follow the greedy policy otherwise - which can converge really slow.


##### Exploration function

The following equation does this:

$$
U^+(s) \leftarrow R(s) + max_a f \Big{(} 
sum_{s'} P(s | s, a) U^+(s), N(s, a) \Big{)}
$$

here, $f(u, n)$ is called the **exploration function**. It determines how greed is traded off against curiosity. One particular simple definition is

$$
f(u, n) = R^+ if n < N_e, otherwise u
$$

where $R^+$ is an optimistic estimate of the best possible reward obtainable in any state, $U^+(s)$ be the optimistic estimate of the utility (reward-to-go), and $N_e$ is a fixed parameter.


#### Q-learning

There is an alternative TD method, called **Q-learning**, which learns an action-utility representation instead of learning utilities.

We will use the notation $Q(s, a)$ to denote the value of doing action a in state $s$. Q-values are directly related to utility values as follows:

$$
U(s) = max_a Q(s,a)
$$

A TD agent that learns a Q-function does not need a model of the form $P(s | s, a)$, either for learning or for action selection. For this reason, Q-learning is called a model-free method.

Q-learning has a close relative called *SARSA* (for State-Action-Reward-State-Action).

The difference from Q-learning is quite subtle: whereas Q-learning backs up the best Q-value from the state reached in the observed transition, SARSA waits until an action is actually taken and backs up the Q-value for that action.




## Chapter 26 - Philosophical Foundations

First, some terminology: the assertion that machines could act as *if* they were intelligent is called the weak AI hypothesis by philosophers, and the assertion that machines that do so are *actually* thinking (not just simulating thinking) is called the strong AI hypothesis.



### Weak AI: Can Machines Act Intelligently?

It is said that: “Every aspect of learning or any other featureof intelligence can be so precisely described that a machine can be made to simulate it.” Thus, AI was founded on the assumption that weak AI is possible.

#### The argument from disability

The "argument from disability" makes the claim that "a machine can never do X". As examples of X, Turing lists the following: 

* Be kind, resourceful, beautiful, friendly, have initiative, have a sense of humor, tell right from wrong, make mistakes, fall in love, enjoy strawberries and cream, make someone fall in love with it, learn from experience, use words properly, be the subject of its own thought, have as much diversity of behavior as man, do something really new.

#### The argument from informality


One of the most influential and persistent criticisms of AI as an enterprise raised by Turing as the “argument from informality of behavior.” Essentially, this is the claim that human behavior is far too complex to be captured by any simple set of rules and that because computers can do no more than follow a set of rules, they cannot generate behavior as intelligent as that of humans.

The inability to capture everything in a set of logical rules is called the qualification problem in AI.



To understand how human (or other animal) agents work, we have to considerthe whole agent, not just the agent program. Indeed, the embodied cognition approach claims that it makes no sense to consider the brain separately: cognition takes place within a body, which is embedded in an environment.


#### Strong AI: Can machines really think?


Many philosophers have claimed that a machine that passes the Turing Test would still not be actually thinking, but would be only a simulation of thinking.

One can easily imagine some future time in which complex conversations with machines are commonplace, and it becomes customary to make no linguistic distinction between “real” and “artificial” thinking.

Are mental processes more like storms, or more like addition? Turing’s answer—the polite convention suggests that the issue will eventually go away by itself once machines reach a certain level of sophistication. This would have the effect of dissolving the difference between weak and strong AI. Against this, one may insist that there is a factual issue at stake: humans do have real minds, and machines might or might not. To address this factual issue, we need to understand how it is that humans have real minds, not just bodies that generate neurophysiological processes. Philosophical efforts to solve this mind–body problem are directly relevant to the question of whether machines could have real minds.


The mind–body problem faced by *dualists* is the question of how the mind can control the body if the two are really separate.

The *monist* theory of mind, often called physicalism, avoids this problem by asserting the mind is not separate from the body — that mental states are physical states.

##### Mental states and the brain in a vat

Physicalist philosophers have attempted to explicate what it means to say that a person—and, by extension, a computer—is in a particular mental state.

They have focused in particular onintentional states.These are states, such as believing, knowing, desiring, fearing, and so on, that refer to some aspect of the external world.

The simplicity of this view is challenged by some simple thought experiments. Imagine, if you will, that your brain was removed from your body at birth and placed in a marvelously engineered vat. The vat sustains your brain, allowing it to grow and develop. At the same time, electronic signals are fed to your brain from a computer simulation of an entirely fictitious world, and motor signals from your brain are intercepted and used to modify the simulation as appropriate.


One could say that the content of mental states can be interpreted from two different points of view:

* **“Wide content”** view interprets it from the point of view of an omniscient outside observer with access to the whole situation, who can distinguish differences in the world. Under this view, the content of mental states involves both the brain state and the environment history.

* Narrow content, on the other hand, considers only the brain state. The narrow content of the brain states of a real hamburger-eater and a brain-in-a-vat “hamburger”-“eater” is the same in both cases.


**Functionalism:**

The theory of *functionalism* says that a mental state is any intermediate causal condition between input and output. This implies that a computer program could have the same mental states as a person.


A strong challenge to functionalism has been mounted by J. Searle’s ('80) **biological naturalism**, according to which mental states are high-level emergent features that are caused by low-level physical processes in the neurons, and it is the (unspecified) properties of the neurons that matter. Thus, mental states cannot be duplicated just on the basis of some program having the same functional structure with the same input–output behavior; we would require that the program be running on an architecture with the same causal power as neurons.


**Consciousness:**

Running through all the debates about strong AI—the elephant in the debating room, so to speak—is the issue of **consciousness**. Consciousness is often broken down into aspects such as understanding and self-awareness.


Consider, for example, the inverted spectrum thought experiment, which the subjective experience of person X when seeing red objects is the same experience that the rest of us experience when seeing green objects, and vice versa. X still calls red objects “red,” stops for red traffic lights, and agrees that the redness of red traffic lights is a more intense red than the redness of the setting sun. Yet, X’s subjective experience is just different.


Suppose, for the sake of argument, that we have completed the process of scientific research on the brain—we have found that neural process P12 in neuron N177 transforms molecule A into molecule B, and so on, and on. There is simply no currently accepted form of reasoning that would lead from such findings to the conclusion that the entity owning those neurons has any particular subjective experience. This **explanatory gap** has led some philosophers to conclude that humans are simply incapable of forming a proper understanding of their own consciousness.


#### The ethics and risks of developing Artificial Intelligence 

Problems:
* People might lose their jobs to automation.
* People might have too much (or too little) leisure time.
* People might lose their sense of being unique.
* AI systems might be used toward undesirable ends.
* The use of AI systems might result in a loss of accountability.
* The success of AI might mean the end of the human race.


**Ultraintelligent machine:**

Let an **ultraintelligent machine** be defined as a machine that can far surpass all the intellectual activities of any man however clever. Since the design of machines is one of these intellectual activities, an ultraintelligent machine could design even better machines; there would then unquestionably be an “*intelligence explosion*,” and the intelligence of man would be left far behind. Thus the first ultraintelligent machine is the last invention that man need ever make, provided that the machine is docile enough to tell us how to keep it under control.


##### Summary

Philosophers use the term weak AI for the hypothesis that machines could possibly behave intelligently, and strong AI for the hypothesis that such machines would count as having actual minds (as opposed to simulated minds).



## Chapter 27 - The Present and Future



#### Agent Components

*Interaction with the environment through sensors and actuators:* For much of the history of AI, this has been a glaring weak point.


*Keeping track of the state of the world:* This is one of the core capabilities required for an intelligent agent. It requires both perception and updating of internal representations.


*Projecting, evaluating, and selecting future courses of action:* The basic knowledge-representation requirements here are the same as for keeping track of the world; the primary difficulty is coping with courses of action—such as having a conversation or a cup of tea —that consist eventually of thousands or millions of primitive steps for a real agent. It is only by imposing hierarchical structure on behavior that we humans cope at all.

*Utility as an expression of preferences:* In principle, basing rational decisions on the maximization of expected utility is completely general and avoids many of the problems of purely goal-based approaches, such as conflicting goals and uncertain attainment.

*Learning:* The chapters above describes how learning in an agent can be formulated as inductive learning (supervised, unsupervised, or reinforcement-based) of the functions that constitute the various components of the agent.



##### Possible ways to go with AI:

**Perfect rationality:**

A perfectly rational agent acts at every instant in such a way as to maximize its expected utility, given the information it has acquired from the environment. Wehave seen that the calculations necessary to achieve perfect rationality in most environments are too time consuming, so perfect rationality is not a realistic goal.

**Calculative rationality:**

A calculatively rational agent eventually returns what would have been the rational choice at the beginning of its deliberation.

**Bounded optimality (BO)**:

A bounded optimal agent behaves as well as possible, given its computational resources. That is, the expected utility of the agent program for a bounded optimal agent is at least as high as the expected utility of any other agent program running on the same machine.






## Paper: Case-Based Reasoning: Foundational Issues, Methodological Variations, and System Approaches


Instead of relying solely on general knowledge of a problem domain, or making associations along generalized relationships between problem descriptors and conclusions, CBR is able to utilize the specific knowledge of previously experienced, concrete problem situations (cases). 

A new problem is solved by finding a similar past case, and reusing it in the new problem situation. 

A second important difference is that CBR also is an approach to incremental, sustained learning, since a new experience is retained each time a problem has been solved, making it immediately available for future problems.


Let us illustrate this by looking at some typical problem solving situation:

*A physician* - after having examined a particular patient in his office - gets a reminding to a patient that he treated two weeks ago. Assuming that the reminding was caused by a similarity of important symptoms (and not the patient's hair color, say), the physician uses the diagnosis and treatment of the previous patient to determine the disease and treatment for the patient in front of him.



As the above example indicate, reasoning by re-using past cases is a powerful and frequently applied way to solve problems for humans. 


In CBR terminology, a case usually denotes a problem situation. 

##### Learning 

Learning in CBR occurs as a natural by-product of problem solving. When a problem is successfully solved, the experience is retained in order to solve similar problems in the future. 

When an attempt to solve a problem fails, the reason for the failure is identified and remembered in order to avoid the same mistake in the future. 


#### CBR Methods


##### Exemplar-based reasoning

In the exemplar view, a concept is defined extensionally, as the set of its exemplars. CBR methods that address the learning of concept definitions (i.e. the problem addressed by most of the research in machine learning), are sometimes referred to as exemplar-based. 


##### Instance-based reasoning

This is a specialization of exemplar-based reasoning into a highly syntactic CBR-approach.

To compensate for lack of guidance from general background knowledge, a relatively large number of instances are needed in order to close in on a concept definition. 


##### Memory-based reasoning

This approach emphasizes a collection of cases as a large memory, and reasoning as a process of accessing and searching in this memory. 

The utilization of parallel processing techniques is a characteristic of these methods, and distinguishes this approach from the others. 

##### Case-based reasoning

First, a typical case is usually assumed to have a certain degree of richness of information contained in it, and a certain complexity with respect to its internal organization. 

That is, a feature vector holding some values and a corresponding class is not what we would call a typical case description. What we refer to as typical case-based methods also has another characteristic property: They are able to modify, or adapt, a retrieved solution when applied in a different problem solving context.


##### Analogy-based reasoning.

It is also often used to characterize methods that solve new problems based on past cases from a different domain, while typical case-based methods focus on indexing and matching strategies for single-domain cases. Re-search on analogy reasoning is therefore a subfield concerned with mechanisms for identification and utilization of cross-domain analogies.

Big focus on *reuse* of past cases. 


#### The CBR Cycle

1. **Retrieve** the most similar case or cases
2. **Reuse** the information and knowledge in that case to solve the problem
3. **Revise** the proposed solution
4. **Retain** the parts of this experience likely to be useful for future problem solving


A new problem is solved by *retrieving* one or more previously experienced cases, *reusing* the case in one way or another, *revising* the solution based on reusing a previous case, and *retaining* the new experience by incorporating it into the existing knowledge-base (case-base). 

An initial description of a problem (top of Fig. 1) defines a new case. This new case is used to RE-TRIEVE a case from the collection of previous cases. The retrieved case is combined with the new case - through REUSE - into a solved case, i.e. a proposed solution to the initial problem. Through the REVISE process this solution is tested for success, e.g. by being applied to the real world environment or evaluated by a teacher, and repaired if failed. During RETAIN, useful experience is retained for future reuse, and the case base is updated by a new *learned case*, or by modification of some existing cases.


#### Representation of Cases

A case-based reasoner is heavily dependent on the structure and content of its collection of cases - often referred to as its case memory.


**The Dynamic Memory Model:**

The basic idea is to organize specific cases that share similar properties under a more general structure (a generalized episode - GE). A generalized episode contains three different types of objects: Norms, cases and indices.

Norms are features common to all cases indexed under a GE. Indices are features that discriminate between a GE's cases.


If - during the storage of a case - two cases (or two GEs) end up under the same index, a new generalized episode is automatically created. Hence, the memory structure is dynamic in the sense that similar parts of two case descriptions are dynamically generalized into a GE, and the cases are indexed under this GE by their difference features.


Any attempt to generalize a set of cases should - if attempted at all - be done very cautiously. 


##### Case Retrieval 

The Retrieve task starts with a (partial) problem description, and ends when a best matching previous case has been found. Its subtasks are referred to as

* Identify Features
* Initially Match
* Search
* Select

##### Case Reuse

The reuse of the retrieved case solution in the context of the new case focuses on two aspects: (a) the differences among the past and the current case and (b) what part of retrieved case can be transferred to the new case. The possible subtasks of Reuse are *Copy* and *Adapt*.

##### Case Revision

When a case solution generated by the reuse phase is not correct, an opportunity for learning from failure arises. 

If successful, learn from the success (case retainment, see next section), otherwise repair the case solution using domain-specific knowledge or user input.


##### Case Retainment - Learning

This is the process of incorporating what is useful to retain from the new problem solving episode into the existing knowledge. The learning from success or failure of the proposed solution is triggered by the outcome of the evaluation and possible repair.

It involves selecting which information from the case to retain, in what form to retain it, how to index the case for later retrieval from similar problems, and how to integrate the new case in the memory structure.

* Extract
* Index
* Integrate


































