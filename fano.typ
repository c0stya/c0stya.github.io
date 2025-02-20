#link("./index.html")[Home] | #link("./about.html")[About]

= Fano's inequality or why your features have to be informative

What does it mean to extract a good feature? Part of the answer lies in ensuring the feature is informative. But how much information should it contain? We can formally estimate this using *Fano's inequality*:

$
H(X|Y) <= H_b(epsilon) + epsilon dot.c log(|C|-1)
$

where

$
H_b(epsilon) = -epsilon dot.c log(epsilon) - (1 - epsilon) dot.c log(1 - epsilon)
$

is the *binary entropy*, $C$ is a random variable representing the *classifier* and $|C|$ is a number of classes.

Let's break this down. Let $X$ be an $n$-ary random variable we want to predict. Let $Y$ be a random variable representing the features we feed into our classifier. We build a classifier $C$ to be as close as we can to match the hidden label $X$. Suppose we aim for a small error rate, not exceeding $epsilon$. Formally, $P(X=C) = 1 - epsilon $.

Let's rewrite the original inequality (\*) to be suitable for our needs. By definition of the conditional entropy:

$
H(X|Y) = H(X) - I(X,Y)
$

substituting the above into (\*) inequality we have:

$
H(X) - I(X,Y) <= H_b(epsilon) + epsilon dot.c log(|C|-1)
$
then,
$
I(X,Y) >= H(X) - H_b(epsilon) - epsilon dot.c log(|C|-1)
$

In plain English it means that in order to have good classification quality $1-epsilon$. we need our features to share at least $H(X) - H_b(epsilon) - epsilon dot.c log(|C|-1)$ bits with the original data $X$.

== Example

Lets take an example of a classification problem with 10 classes. It can be a MNIST dataset for example. Assume it has $|C|=10$ classes uniformely distributed. That means $H(X) = log(10)$ of total information. Assume we want to design a classifier with error at least $0.9$, then

$
I(X,Y) 	& >= H(X) - H_b(epsilon) - epsilon dot.c log(|C|-1) \
	& = H(10) - H_b(0.1) - 0.1 \cdot log(10-1) \
	& = log(10) - (0.1 dot.c log(0.1) + 0.9 dot.c log(0.9)) - 0.1 dot.c log(9) \
	& approx 2.54 "bits"
$

Thus, to achieve 90% accuracy, we need at least $2.54$ bits of class information per MNIST image. Compared to the original entropy of $log(10) approx 3.32$ bits, this is about 76% of the class information.

If we require the error to be zero (i.e. $epsilon = 0$), then inequality simplifies to:

$
I(X,Y) 	& >= H(X) - H_b(epsilon) - epsilon dot.c log(|C|-1) \
	& = H(X)  \
	& = log(10) \
	& = 3.32 \
$

That means we have to keep all the information in our. It does agree with our intuition. To have the perfect prediction we have to have all the information.

Opposite, if we don't care about the error and set it to 0.1 (i.e. $epsilon$=0.1, which corresponds accuracy of 0.5 for 10 classes).

$
I(X,Y)>= H(X) - H_b(epsilon) - epsilon dot.c log(|C|-1) = 0
$

So, again, it aligns with our intuition. To produce random predictions we don't have to have information at all.

== Mutual information is not enough

We've shown that good classifiers require informative features - features that share a significant amount of information with the original data $X$. However, this is only an *upper bound* on classifier quality. Even with highly informative features, we can still end up with a poor classifier.

Consider the case where we build a classifier directly from the *raw data* $X$. While we have all the information, constructing a classifier directly from raw pixels or signals is extremely challenging. This is why feature engineering or automatic feature extraction (as in deep learning) is typically necessary.

Another example is *encrypting* the original data. While the encrypted data retains all the original information, building a model without the encryption key is nearly impossible.

This suggests that while informativeness is necessary, it is not the only requirement for good features. Intuitively, features should also be *simple*. This could mean linear separability, mutual independence, or some form of "disentanglement." I strongly believe that good features should have low computational complexity (e.g., #link("https://en.wikipedia.org/wiki/Kolmogorov_complexity")[Kolmogorov complexity] or the #link("https://en.wikipedia.org/wiki/Minimum_description_length")[minimum description length]. However, this remains an open question in general.
