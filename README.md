# APS360 Project: Applied Fundamentals of Deep Learning 
### EcoSort: Garbage Classification using Pytorch 
---
### Collaborators

Areeba Mobeen  
`areeba.mobeen@mail.utoronto.ca`

Briana Vieira  
`briana.vieira@mail.utoronto.ca`

Rayan Mustafa  
`rayan.mustafa@mail.utoronto.ca`

Hassan Qayyum  
`hassan.qayyum@mail.utoronto.ca`

---

## 1. Introduction

The average person will spend one-third of their lifetime at work and at the same time generate
0.74kg of daily waste, contributing to roughly 2 billion tonnes globally every year (Li & Chen,
2023). Workplaces are challenged with managing a significant amount of waste, our team poses
the question of how successfully this is being done. If executed correctly, waste sorting enables
the recovery and recycling of valuable materials like metals, paper, and plastics, which conserves
natural resources, reduces energy consumption, and lowers greenhouse gas emissions.

Currently, these systems are entirely manual and rely on the consumer to correctly sort their waste,
and are consequently error-prone. As is demonstrated by the millions of tons of garbage that is not
sorted correctly and end up in improper disposal sites, contributing to environmental degradation and
climate change each year (Deer, 2021). In Toronto alone, the blue bin program manages 180,000
tonnes of recyclables annually, yet 30 percent (54,000 tonnes) is sent to landfills due to contami-
nation (Zettler, 2019). This fully manual method has plenty of room for improvement and hinders
progress toward efficient and accurate waste classification by not employing data-driven approaches.
Deep learning, particularly CNNs, offers a promising approach due to its ability to automatically
learn discriminative features from raw data, making it well-suited for image-based tasks like garbage
classification. Other machine learning approaches like RNNs, SVMs, and Random Forests are less
suitable for image-based tasks such as garbage classification. RNNs are designed for sequential data
tasks, like natural language processing or time series analysis, and may not capture the complex spa-
tial patterns in images. SVMs and Random Forests rely on handcrafted features or predefined rules,
which may not capture the diverse visual characteristics of waste materials as effectively. Overall,
CNNs offer a powerful and efficient solution for garbage classification, leveraging the capabilities
of deep learning to automatically learn and extract discriminative features from raw image data
Employing this CNN model approach, our project, EcoSort, addresses the pressing issue of improper
waste management by developing a robust garbage classification system that will facilitate proper
disposal efforts. We plan for our model to classify waste into the following six classes: cardboard,
glass, metal, paper, plastic, and trash, of which the outputs can then be grouped into umbrella cate-
gories as needed. By leveraging the power of deep learning, we aim to enhance waste management
practices, mitigate environmental pollution, and contribute to the fight against climate change.

## 2.0 Background and Related Work

Automation is becoming the norm in many aspects of life, yet it is not used in systems for the
disposal of our daily waste. Let’s take the manual method used at the University of Toronto as
an example, in 2022 the St. George campus only captured 75% of recyclable material, and of the
material placed in recycling, 4% did not belong there (Schwalb, 2023). Contamination poses a major
issue as putting something soiled into the recycling can ruin the rest of the material. The City of
Toronto estimated that “each percentage point decrease in contamination could lower recycling costs
in the city by $600,000 to $1 million a year” (Jonas, 2019).

Currently, U of T is using a 4-bin separation system with signage to guide users on which bins to
place their waste. This system is similar to that of Dalhousie University where a study was done
to look into separation behavior and compliance. It was found that contamination existed in all the
bins with 11.5% of people not sorting their waste correctly, the reasons for which being that people
misunderstood the signs or did not care (Smith et al., 2018). Clearly, there is room for improvement
in this manual system for waste management.

Other groups worked to develop deep learning models that can classify different types or com-
ponents of waste. One student’s thesis from Tampere University was to determine which image
classification model works best to classify trash, with a focus on industrial waste, they achieved

82.2% accuracy using a CNN and found that to be the best model (Bhandari, 2020). This thesis pri-
marily compared CNN to a support vector Machine (SVM), though it also discussed previous work
done in waste image classification. This included Smart Trash Net’s model which classified images
into three classes (disposal, recycling, and paper) and achieved 68% accuracy, ALexNet’s model
which just determined whether images were of trash or not, and SamurAI’s model that recognized
recyclables like cartridges and containers (Bhandari, 2020).

A few other works used a CNN in combination with other methods to create waste classification
models. One of which achieved 95.46% accuracy in municipal solid waste classification using
CNN and Graph long short-term memory. This model classified images into six categories (card-
board, metal, glass, plastic, paper, and organic waste) (Li & Chen, 2023). Two other works used a
pre-trained ResNet model, one of which used a combination of CNN and SVM to make their clas-
sifications (Adedeji & Wang, 2019). The second ResNet model differentiated between waste that is
either biodegradable or not and achieved 98.01% accuracy (Md.Rahman et al., 2022). This paper
also introduced a design for a smart trash bin using a microcontroller and ultrasound sensors. The
bin captures an image of the trash you place inside, determines if it is biodegradable, and then uses
a roller mechanism to move the trash to the correct section and dispose of it (Md.Rahman et al.,
2022).

There is a gap in the literature about a model that classifies waste into the four categories currently
used by U of T (landfill garbage, paper, containers, and coffee cups). Our team was also unsuccessful
in finding previous work done on using a waste classification model to supplement the signs at
garbage bins and help users determine where to correctly place their trash. By learning from and
expanding from the work previously done in this area, our team aims to develop a solution that will
make the waste management systems more successful

## 3. Architecture

The preliminary architecture description below illustrates our envisioned implementation of EcoSort. These serve
as overarching guidelines, providing a strong foundation while allowing for the addition of more
detailed specifics as we progress.

### 3.1 Proposed Architecture Design

EcoSort aims to classify images into six classes of garbage: cardboard, glass, metal, paper, plastic,
and trash. Therefore, a Convolutional Neural Network (CNN) is the most suitable approach. CNNs
excel at feature learning in classification tasks by employing mathematical constructs such as con-
volution filters and pooling. These techniques enable the model to discern distinct and recognizable
features within smaller sections of the data.

Currently, one successful type of CNN is Residual Networks (ResNet). By leveraging Residual
Networks and incorporating skipping connections, we can train a model with a substantial number
of convolutional layers. Skipping connections involve adding the input of the previous layer to the
output of a deeper layer after activation. This technique helps mitigate issues like vanishing and
exploding gradients during backpropagation, as it normalizes changes after each layer.
Additionally, we will employ weight decay and dropout techniques to normalize training, enhance
convergence speed, and prevent overfitting and exploding gradients. Utilizing convolution layers
with 3x3 filters allows us to mimic the effects of larger resolution filters while minimizing the num-
ber of adjustable parameters and weights. This approach reduces training time and enables us to
dedicate more resources to hyperparameter tuning for optimal model configuration.

For activation functions, we will initially use Softmax, which excels in multi-classification problems.
The ReLU function will also be employed as it is generally effective for training neural networks.
Adjusting the activation function during training allows us to maximize accuracy.
By implementing these strategies, we aim to develop a highly effective and accurate CNN model for
garbage classification.

The Figure below presents a high-level architectural concept of the model. Utilizing a CNN, we have a prelim-
inary understanding of its functionality, although specific parameters will require experimentation.
Further elaboration on this is provided in subsequent sections.

<img width="605" alt="image" src="https://github.com/H-Qayyum/Deep-Learning-Project-Garbage-Classification/assets/121575620/b50a7917-d995-42b1-904e-79b45562d42c">  

### 3.2 Model In Use: Concept

The Figure below demonstrates a model instance in operation. Initiated by a camera module capturing input
images, the output is displayed on a screen, facilitating convenient waste disposal for the user.

<img width="416" alt="image" src="https://github.com/H-Qayyum/Deep-Learning-Project-Garbage-Classification/assets/121575620/a442f8ff-8a15-465f-9514-cbefcaaacd9f">

## 4.0 Data Processing

Numerous datasets related to garbage classification are available on Kaggle. Presently, we have ac-
quired three datasets from Kaggle (Unknown, 2019), (Kunwar, 2023), (Mohamed, 2021). Utilizing
a large dataset for training, testing, and validation purposes enables more intensive and accurate
model training. These datasets closely align with our intended classification classes (cardboard,
glass, metal, paper, plastic, and trash). While one dataset perfectly partitions the data into these
classes, two datasets contain additional classes, requiring cleaning to remove irrelevant classes that
may hinder our model’s accuracy. These datasets collectively comprise approximately 20,000 im-
ages, but not all images will be utilized due to irrelevant classes. The data undergoes cleaning to remove inaccuracies, duplicates, and outliers before model training.
Cleaning enhances model accuracy by eliminating biases and redundant data. A Python program
will be developed to identify and remove duplicate data by comparing the tensors of each image.

## 5.0 Ethical Considerations

### 5.1 Ethical Issues
Certain uses of our garbage classification system could give rise to ethical issues such as privacy
concerns and discrepancies in geographical representation in the training data as different areas of
the world have different packaging and waste materials. In addition, images collected for garbage
classification purposes during or after model training could contain personally identifiable informa-
tion about people or their locations, giving rise to privacy risks of breaches. An example of such
garbage is recyclable letters with addresses, labels, and names. In addition, user consent may be
required in case our model captures a person’s face while capturing the item they are discarding.
These instances must be considered to resolve the ethical issue of data retention for the purpose of
improved model training to classify newer garbage data.

### 5.2 Limitation in Training Data

Furthermore, there are several limitations in our training data which lacks representation from differ-
ent geographical regions across the world. This leads to biases in model training as different regions
have a different set of cultural practices, waste management systems, and environmental regulations.
Other limitations exist within our training data such as unbalanced data, in which certain types of
waste, such as garbage, are more prevalent than others. Our model is designed for municipal waste,
and if it is used in unfamiliar regions or in unique scenarios, the classifier will then be inaccurate
and struggle to classify garbage that is not well-represented in the training data. This could lead to
misclassification and incorrect waste management procedures, potentially directing incorrectly clas-
sified garbage into landfills, causing harm to the environment and worsening environmental issues,
such as climate change.

## References

Olugboja Adedeji and Zenghui Wang. Intelligent waste classification system using deep learning convolutional neural network. Elsevier B.V., 2019.  

Pragati Baheti. Train test validation split: How to best practices. V7: Machine Learning, 2021.

Sishir Bhandari. Automatic waste sorting in industrial environments via machine learning approaches. Tampere University, 2020.

Ryan Deer. Landfills: We’re running out of space. ROADRUNNER, 2021.

Goel. Ml — overview of data cleaning. Geeks for Geeks, 2024.

Sabrina Jonas. Garbage or recycling? u of t students invent ’robobin’ to make the decision for you. Canadian Broadcasting Corporation, 2019.

Sakshi Khanna. A comprehensive guide to train-test-validation split in 2024. Analytics Vidhya, 2024.

Suman Kunwar. Garbage dataset, 2023. URL https://www.kaggle.com/ds/2814684. Feb 10, 2024.

Ninghui Li and Yuan Chen. Municipal solid waste classification and real-time detection using deep learning methods. Urban Climate, 2023.

Wahidur Md.Rahman, Rahabul Islam, Arafat Hasan, Nasima Bithi Islam, Mahmodul Md.Hasan,
and Mohammad Motiur Rahman. Intelligent waste management system using deep learning with
iot. Elsevier B.V., 2022.

Mostafa Mohamed. Garbage classification (12 classes), 2021. URL https://www.kaggle. com/datasets/mostafaabla/garbage-classification. Feb 10, 2024.

Jessie Schwalb. The breakdown: Recycling at u of t. The Varsity, 2023.

Laura Smith, Jessica Tam, Phil Martel, and Jeremy Holleran. Waste management at dalhousie university, canada: Social factors affecting recycling behavior and use of the four bin system at the studley campus. 2018.

Unknown. Garbage classification, 2019. URL https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification. Feb 10, 2024.

Melanie Zettler. Toronto recycling: Why so much material still goes to landfill. Global News: Environment, 2019.





