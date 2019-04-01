---
title: "Machine Learning and Object Detection: A Report"
author: "Joseph Borjon"
date: "March 30, 2019"
output:
  html_document:
    keep_md: true
    toc: true
    toc_float: true
    code_folding: hide
    fig_height: 12
    fig_width: 12
    fig_align: "center"
---

# Machine Learning and Object Detection: A Report

## Introduction

This research report explores the applications and potential implications of machine learning technology on the lives of human beings. It is diveded into two parts: Part I delves briefly into the theoretical aspects of the technology, and Part II demonstrates the use of a novel machine learning model, called YOLO, to detect objects in an image or in video footage.

## Part I: Machine Learning

### What is machine learning?

Machine learning is the branch of artificial intelligence that applies algorithms and mathematical models to data in order to perform tasks based on general training, without requiring explicit instructions from a programmer to do so. Typically, machine learning is used to make predictions and inference based on patterns in the existing, or "training," data.

The ideas of machine learning have been around since the 1950s. However, only recently did it become feasible to apply machine learning models to large datasets in a practical way. That's fortunate, since datasets have become too large for humans to analyze on their own.

A few of the most commonly used algorithms for machine learning are:

1. Linear Regression
2. Logistic Regression
3. Linear Discriminant Analysis
4. Classification and Regression Trees
5. Naive Bayes
6. K-Nearest Neighbors
7. Learning Vector Quantization
8. Support Vector Machines
9. Bagging and Random Forest
10. Boosting and AdaBoost

### What are some of the applications of machine learning?

The full potential of machine learning has most likely not been achieved, and many of its applications are probably yet to be discovered. They have the potential to affect all areas of industry and human life. A small sample of current applications follows:

- Predicting user preferences within commercial platforms to enable businessses such as Amazon, Google and Netflix to increase revenue by matching what consumers want.
- Enabling marketers and advertisers to predict their audiences' behaviors for the same purposes.
- Helping to ensure optimal efficiency and compliance in the ever-more regulated financial services industry.
- Spotting hard-to-detect security breaches in secure infrastructures.
- Detecting bulling and hate speech in social media.
- Automated customer support.
- Using machine vision and object detection to enable computers to see, thus creating the possibility for facial security checks, self-driving technology, security camera self-monitoring, detection of undesirable content in media, etc. More on this point in Part II.

Most of the applications mentioned have a tremendous potential for reducing operating costs and improving efficiency, creating opportunities for increased profitability.

### Is machine learning better than the human mind?

This point is debatable and, to a large degree, subjective. How do we define *better?* How do we define *human mind?* Changes and specialization in the technology may render this debate moot in future years.

Humans still produce superior results in areas that require the human touch, such as the arts, counseling, management and studying human behavior and interaction. Where computers excel is where they always have: running large amounts of calculations that would take a human brain considerably longer.

The results achieved by machine learning are still not perfect, as you will see in the discussion in Part II.I some areas, such as healthcare, computers are still struggling to deliver satisfactory results in spite of billions of dollars of investment.

### What are the problems we need to solve?

Because computers can't make human judgements, the *garbage in, garbage out* principle needs to be heavily considered. In other words, biased training data will result in biased predictions from the model. Human language is inherently biased, so all language-related ML models will have some degree of bias.

Another potential issue is the danger of intentional bias put into machine learning models by stakeholders who stand to benefit from additional sales if the undisclosed, proprietary models predict or overstate needs that don’t actually exist. Therefore, algorithms need to be transparent to inspection.

A third potential issue is the possibility of machine learning predictions becoming so accurate and generalized that we are able to predict with great certainty such things as likely deadly diseases in an individual’s genome or a detailed probable path of economic activity throughout a person’s lifespan. What ill-intentioned organizations may be able to do with such data is potentially devastating.

### An expert's optimism---and a word of caution

In March 2019, Stanford's [Dr. Fei-Fei Li](http://vision.stanford.edu/feifeili), who is largely responsible for the current state of machine learning, and particularly machine vision, sat down with *Wired* for an [interview](https://www.wired.com/story/fei-fei-li-ai-care-more-about-humans). In it, she spoke of her worry that AI "may not always make the world better" and of her plan to make the technology more human-centric.

She gave the example of an ICU, where AI could help relieve the 24/7 monitoring that overworked medical professionals must perform, without replacing the exclusively human functions those professionals provide.

She also made the point that, while the public often associates AI technologies with the tech industry, the true aim of AI researchers is to extend it to improve "manufacturing, agriculture, retail, health care, education, government," and virtually all areas of human activity.

"There is a big role for AI to play in terms of helping the world in many important issues," she
said, “but we have to guide it in the most thoughtful and human-centric way."

## Part II: Object Detection and YOLO

### What is object detection?

Object detection is an application of machine learning through which specific objects, animals or people can be identified within any digital image. As with any machine learning algorithm, object detection requires large amounts of training data with known objects in order to learn how to recognize the patterns that make up that object.

Older computer vision algorithms were bound by the heuristics that developers were able to think of and build into the system---they were not satisfactorily accurate nor could they handle objects that didn't fit the built-in parameters. Through years of effort in applying convolutional neural networks (CNNs) to the problem of object recognition and then assembling a gigantic training dataset with workers all over the world, Dr. Fei Fei Li and her team were able to teach machines how to see on their own, figuratively speaking.

Given that video is merely a sequence of still images, object recongition is equally applicable to static images and video. Models scan each video frame and look for objects there, then they move to the next.

### What is YOLO?

YOLO stands for *You Only Look Once.* It is an optimized object-recognition method.

[Joseph Redmon](https://www.youtube.com/watch?v=Cgxsv1riJhI), of the University of Washington, explained how other methods do their work (essentially, they scan the whole image several times for each object) and how his team was able to examine the whole image in a single pass instead---hence, *You Only Look Once.* The CNN-based YOLO method brings the time required to detect and recognize objects down by a tremendous factor, enabling even low-grade hardware to run simplified versions of the model at an acceptable speed while decreasing the cost of the technology, bringing it to the masses. Because YOLO is free and open source, anyone can use it for any purpose.

### What are the mechanics of YOLO?

![Dog](https://machinethink.net/images/yolo/Grid@2x.png "Doggity dog-dog")

![Dog v2](https://machinethink.net/images/yolo/Grid@2.png "Doggity dog-doggo")

### YOLOv2 demo

In order to experiment with the technology, I decided to download and apply a Python/TensorFlow version of YOLOv2 (v3, which is faster, is already out but there was much more training material available for v2). To keep things simple, I used the training weights already provided, though it seems plausible that the accuracy of the model could have been improved by using a larger, more varied dataset.

In the resulting video below, you'll notice that clearly visible objects are misidentified or not identified at all. This is expected. Given my hardware and time constraints, I used a simplified version of YOLO, called Tiny YOLO, which uses fewer layers and is, therefore, faster but less accurate. It would be interesting to re-run the model using full-sized YOLO.

Here's the demo. Enjoy.

<iframe width="560" height="315" src="https://www.youtube.com/embed/ruDXYYldV1E" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Sources

- https://en.wikipedia.org/wiki/Machine_learning
- https://en.wikipedia.org/wiki/Timeline_of_machine_learning
- https://www.sas.com/en_us/insights/analytics/machine-learning.html
- https://www.sas.com/en_gb/insights/articles/analytics/applications-of-machine-learning.html
- https://towardsdatascience.com/a-tour-of-the-top-10-algorithms-for-machine-learning-newbies-dde4edffae11
- https://medium.com/app-affairs/9-applications-of-machine-learning-from-day-to-day-life-112a47a429d0
- https://intelligence.org/files/EthicsofAI.pdf
- https://www.wired.com/story/fei-fei-li-ai-care-more-about-humans
- https://www.youtube.com/watch?v=40riCqvRoMs
- https://www.youtube.com/watch?v=Cgxsv1riJhI
- https://machinethink.net/blog/object-detection-with-yolo/#how-yolo-works
