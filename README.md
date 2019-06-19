# Introduction to nnvc

NNVC - Neural Network Video Compression Project
-----------------------------------------------

NNVC was born out of a research project I worked on in school studying Neural Networks
built and trained by Genetic Algorithms. It nagged at me that there had to be a practical
application for these advanced techniques. After studying Neural Networks for so long,
their various properties congealed into an amorphous idea that they could be used for
data compression. I chose video compression in particular because it had the most
interesting reprecussions in implementation and industry.

Video
-----
Digital Video in practice is a well studied discipline. It is rather straightforward in
reality. Videos are a series of frames of pictures made up of pixels, each with a
Red, Green, Blue, and Alpha integer value to produce color. These properties make video
easy to study. The problem with video is that any substantial quantity of it is large.
Modern practice for transferring video on demand is to stream it frame by frame
across networks. This causes frightful load on networks considering how much video is
being streamed at any given moment across networks: Petabytes of the stuff! If the
transfer mechanism of video were to be converted to traditional file transfer, load
on internet networks would fall radically. Thus, compression is a means to an end in
this project. The ultimate goal is to fundamentally change video transfer over the
internet. If video were to be compressed to a drastic degree before sending and then
dynamically reproduced on the other side of the transfer then network load would fall,
bandwidth requirements would shrivel, and the cost of streaming, which is substantial to
everyone involved, would virtually disapper. The key is to be able to compress video to
such a degree as to make this feasible.

Neural Network Compression
--------------------------
To think about compression through Neural Networks requires some abstract thinking.
Mathematically, Neural Networks are considered Universal Approximation Functions. In
other words, they have the ability to map themselves from some arbitrary input to
some arbitrary output. Notably, these inputs and outputs exist in vector space.
Take note that here is where I depart into my own analysis. Networks are dimensional.
They have in interface into vector space that is flexible on both ends, both input
and output. If one holds this true, networks may map low dimensional vectors onto
high dimensional vectors and visa versa. Here is where we find the first property
of Network Compression:

      1. **Neural Networks may compress data through their function**

Take a vector A of length 1 and a vector B of length N. If the Network N is trained
to approximate a mapping between the input A and the output B, then it could be
said that a single feed-forward operation by that network N decompresses vector A
into vector B. Thus, it can be said that the inverse operation, the training of
the network, is by definition a compression operation. Thus, Networks may compress high
dimensional data onto low dimensional data.

However, this is only one stage of compression. Networks have other properties that
prove that they compress trained data. The second property of Network Compression:

      2. **Neural Networks may compress data through their repeated operation**

First, construct a Network N of a finite and subsequently static size. Next, take
a set of vectors A whose members are of the same abitrary length and is itself of
a finite size and then take another set of vectors B whose members are of the
same arbitrary length and is itself of a finite size. Now, grow both A and B such
that the total data contained in the vectors is greater than the size of the
Network N. Train N to map the members of A onto the members of B. Now it can
be said that because the total size of the data N has been trained on is bigger
than N itself, the repeated feed-forward operation of N on A to reproduce B
is in effect decompressing data out of the durable memory of the Network N. Thus
it can be said again that the inverse operation, training, is by definition
a compression operation.

This property is very powerful due to the property of Neural Networks to store
very large amounts of training data. Thus the amount of data that can be stored
in a Network can be many times larger than the Network itself, potentially by
a huge margin.

There is one final property of Network Compression:

    3. **Neural Networks my compress data through their ability to produce other**
     **Neural Networks**

Neural Networks can be expressed in vector form. Saving the weights of a network
as matrices allows for high performance processing. However, this also allows
for another special property that is not immediately obvious, yet is very
powerful. If a Neural Network is comprised of nothing but vectors of weights,
these vectors of weights can be approximated by another Neural Network. To put
this as simply as possible: Neural Networks can produce other Neural Networks.

Take a Network A that is comprised of some large number of weights AW. Then
specify a Network B that is comprised of far fewer weights BW. Network B can
be trained to produce the weights of Network A. Of course, there are technical
issues that will be difficult to overcome. Network weights are extremely
precise and approximating the weights of Network A with another Network
introduces the possiblity for error. Secondly, it is difficult to say without
much experimentation what the specifications for Network B should be.
Difficulties aside, this property is essential for going the extra mile in the
compression schema.