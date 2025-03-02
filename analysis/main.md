# Recap and background
Here we recap the findings we have and review some background on  continual learning from the point of view of Ricahrd Sutton. 


*Broad objectives*: Through a series of works on Richard Sutton, he proposes that one of the most fundamental problems facing AI today is that we cannot let a model continually learn different tasks. This problem has two facets: 1) the catastrophic forgetting, which is forgetting what network has learned in the past, and 2) the issue of loss of plasticity, which is the network losing its ability to learn new concepts. We focus on the later issue of CL her, as it clearly relates to the theoretical result proven later. While there are numerous valuable observations and remedies in these works, there are numerous gaps in our understanding of CL, which makes us unable to build models and algorithms that are able to achieve CL in the same fashion as humans or animals in the real world achieve. Our attempt is to give the key ideas and notions in continual learning a theoretical footing, thereby consolidating the key root causes of loss plasticity. 

We first start by a review of loss of plasticity from Richard Sutton’s point of view, a review of hyper cloning. Then, we elaborate on our theories regarding the noise-less and noisy cases. Finally, we draw on parallels between the key results of our theory and loss of plasticity, where we highlight where it aligns and where it might offer a different perspective on fundamental causes of loss of plasticity.  Based on these theoretical results, we make several hypotheses on how these root causes of loss of plasticity emerge during prolonged training. We would like to understand the architectural and training dynamics that contribute to loss of plasticity in order to both avoid the loss of plasticity during training, and be able to recover from it if it is not catastrophic. Finally, we will also rely on empirical evidence to evaluate the hypotheses that were inspired by our theory. 

## Richard Sutton's view on Continual Learning (CL):
 Here we very briefly review the key notions and ideas that are proposed by Richard Sutton to diagnose, quantify and alleviate barriers to continual learning. Through a series of works, he observes that after having being trained one some tasks, the model’s ability to learn new tasks drops irrecoverably the level of a shallow or even linear model.  Here’s a brief recap of his key contributions to understanding and remedies for loss of plasticity. 

1. Standard Deep Learning and Continual Learning:
    * One‐Time vs. Continual Learning: Traditional deep‐learning methods (using backpropagation with gradient descent or variants such as Adam) are designed for “one‐time” training on a fixed dataset. In many real‐world applications—such as robotics, streaming data, or online reinforcement learning—the data distribution changes over time, requiring the network to continually learn.
    * Loss of Plasticity: Over time, as standard training continues in a non-stationary (continual) learning setting, deep networks lose their “plasticity” (i.e. the ability to quickly adapt to new data). This loss is manifested in several ways:
        * The weights tend to grow larger.
        * A growing fraction of neurons become “dead” (or saturated), meaning that they rarely change their output.
        * The internal representations (the “feature diversity”) become less rich, as measured by a decrease in the effective rank of the hidden layers.
    * This degradation means that—even if early performance on new tasks is good—the network eventually learns no better than a shallow (or even a linear) system when faced with many successive tasks.
2. Empirical Demonstrations:
    * Extensive experiments were conducted on supervised tasks (e.g., variations of ImageNet, class-incremental CIFAR‑100, Online Permuted MNIST, and a “Slowly Changing Regression” problem) and reinforcement learning tasks (such as controlling an “Ant” robot with changing friction).
    * In all these settings, standard backpropagation methods initially learn well but then gradually “forget how to learn” (i.e. they lose plasticity) over hundreds or thousands of tasks.
3. Maintaining Plasticity by Injecting Randomness:
    * The initial random weight initialization provides many advantages (diverse features, small weights, non-saturation) that enable rapid learning early on. However, because standard backprop only applies this “randomness” at the start, these beneficial properties fade with continued training.
    * The key idea is that continual learning requires a sustained injection of randomness or variability to maintain plasticity.
4. Continual Backpropagation (CBP):
    * To counteract the decay of plasticity, the authors propose an algorithm called Continual Backpropagation. CBP is almost identical to standard backpropagation except that, on every update, it selectively reinitializes a very small fraction of the network’s units.
    * Selective Reinitialization: Using a “utility measure” that assesses how useful a neuron (or feature) is for the current task (based on factors such as its activation, its outgoing weight magnitudes, and how much it is changing), the algorithm identifies neurons that are “underused” or “dead.” These neurons are then reinitialized (with the initial small random values), thereby reintroducing diversity and the benefits of a fresh start.
    * This process—sometimes called a “generate-and-test” mechanism—allows the network to continually inject new random features without having to completely reset or lose past learning.
5. Comparison with Other Methods:
    * Other techniques such as L2 regularization, Shrink and Perturb (which combines weight shrinkage with noise injection), dropout, and normalization were examined.
    * Although L2 regularization and Shrink and Perturb help slow the growth of weights and partially mitigate the loss of plasticity, they are generally less robust than CBP. In some experiments (both in supervised and reinforcement learning settings), popular methods like Adam (with standard parameters), dropout, and even batch normalization actually worsened the loss of plasticity over time.
6. Implications for Continual and Reinforcement Learning:
    * The findings imply that if deep neural networks are to be deployed in environments where continual adaptation is necessary, the training algorithms must be modified to continuously “refresh” the network’s ability to learn.
    * In reinforcement learning, where both the environment and the agent’s behavior can change over time, the loss of plasticity is especially problematic. The continual backpropagation approach (sometimes combined with a small amount of L2 regularization) was shown to significantly improve performance in nonstationary RL tasks (for example, in controlling an ant robot in environments with changing friction).
7. Broader Perspective:
    * The work challenges the assumption that gradient descent alone is sufficient for deep learning in dynamic, nonstationary settings.
    * It suggests that “sustained deep learning” (learning that continues to adapt over time) may require algorithms that combine traditional gradient-based methods with mechanisms for continual variability—in effect, a built-in “refresh” mechanism similar to how biological systems continually reorganize and adapt their neural circuitry.


## Hyper cloning recap and connection to loss of plasticity 

 In hyper cloning, authors showed various methods to “enlarge” a trained smaller model, such that its forward pass is perfectly preserved during cloning process, and showing that it can be trained much faster than a large model from scratch while achieving a similar accuracy. However, authors also observe that in the cloning strategies that were “noiseless” the training was not as effective, and had to introduce some types of noise to ensure the training diverged from a simple setting. 

*The theoretical result on cloning:* In this work, we first prove a series of result that are directly stated in terms of the 
We proved in the draft that if we duplicate a hyper-clone a model in a super-symmetric way (more accurately, forward and backward symmetry hold), then the forward and backward vectors of the network are cloned. More concretely, the forward and backward of the model are essentially the cloned (duplicated) versions of a smaller model from which they are cloned. This situation has a very dramatic consequence that, we can perfectly predict the training dynamics of the larger model with a smaller model, with the only caveat that the learning rate for different layers are set in a layer and module-dependent manner.  On the other hand, we show that the noisy cloning strategies can be modeled as the noiseless close plus some additive noise to the backprop gradients, where norm of the gradients per each layer may depend on their depth and network parameters in general. 

## Connection to loss of plasticity?

From a high level point of view, we can view the noiseless cloning strategies as analogues of normal backprop, in that this particular way of cloning may catastrophically limit the model's ability to learn, because it will always be as good as the smaller model, which highly resembles  notion of loss of plasticity phenomenon. 
More concretely, the fact that we can prove that the forward representation remain cloned throughout training, also implies indirectly that the rank of forward representations the larger model will always be equal to rank of the smaller model. This is arguably a very strong case of loss of rank (hard rank as opposed to empirical rank in Richard Sutton’s paper), which is provably unrecoverable! 


*Recovery of loss of plasticity*: Furthermore, our theory shows that  noisy cloning strategies as analogues of continual backprop in that they can be viewed as a normal backprop plus some injected noise.  Therefore, the ability to recover the large model capacity by having some type of noise is an analogue of resetting certain weights and units in continual backprop. 

*Key differences and remaining questions* Despite the similarity of our results on key aspects, there remains some important differences and open questions. 

Firstly, if we assume that in the smaller model all neurons are active and the weights are not saturated, the larger model will have similar properties. Therefore, we construct examples of a network that have catastrophic loss of plasticity, while having no dead neurons or overly large weights. In other words, our theory suggests that while dead and saturated neurons may be a sufficient condition for loss o plasticity, they are not necessary conditions. Conversely, the examples observed by Richard Sutton and other empirical evidence suggests that a network can experience loss of plasticity without having explicitly cloned neurons. In yet another words, while cloning or dead or saturated neurons may occur in some cases of loss of plasticity, but not all cases. In our quest to understand loss of plasticity, we will be searching for a set of “universal” conditions that are necessary and sufficient. Another key aspect of loss of plasticity is that without direct intervention, they will “persist” throughout training. Thus, we must be able to demonstrate  that once these conditions  are met, they will persist to be true throughout training, unless  something like a noise injection breaks these properties explicitly. We can refer to these properties as “canonical properties” of loss of plasticity. 

Secondly, even if we show that these conditions are universal and persistent, it is unclear how these conditions “emerge” during training? In other words, even if we have correctly identified these universal causes and can prove they are persistent, it’s still highly non-trivial why and how these conditions emerge as a result of prolonged training.  For example, while dead neurons or saturated neurons or cloned neurons , it is not guaranteed at all (theoretically or empirically) at all if such properties will emerge as a result of prolonged training. 
Since we know that certain choices of models and training will improve or deteriorate the emergence of these conditions, one can argue that that answer to this “emergence” question will be architecture and configuration-dependent  and not universal. Thus, once we can establish some clear links between training dynamics and model architecture, and loss of plasticity, we can also suggest ways to avoid loss of plasticity, or suggest methods to recover from it. 

Thus, these are the following key questions regarding loss of plasticity we are trying address in this work:
* *Canonical conditions*: What are they underlying universal conditions  that are necessary and sufficient, and they will persist throughout training?  
* *Emergence with training*: How do these conditions emerge as a result of prolonged training? “

### Canonical conditions
Let us try to capitalize on our earlier theories about cloning to arrive at a formalization of the question we are trying to address: 
 *Definition*: In a neural network with parameters $\Theta$ , we define:
* Universal: we say $C(\Theta)$ a universal condition for loss of plasticity for network, if there some smaller network  with parameters $\theta$ ($|Theta| > |\theta|$), such that the larger model hidden units are exactly a copy of the smaller model up to some duplications. 
* Persistent:  having the we say that a $C(\Theta)$ is persistent
Let’s start by reviewing the key aspect of our theory on cloning and Richard Sutton’s views on loss of plasticity. Most notably, both views suggest that  a collapse of forward representations is a key indicator of loss of plasticity. However, as mentioned earlier, neither dead, saturated, nor neurons nor duplicated are necessary for loss of plasticity to occur. 




