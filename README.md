## Multimodal Hate Speech Detection from Bengali Memes and Texts
This repository contains the Multimodal Bengali Hate Speech Dataset, data collection and annotation process, and supplementary information for our paper "Multimodal Hate Speech Detection from Bengali Memes and Texts" submitted to International conference on Speech & Language Technology for Low-resource Languages (SPELLL).  
This dataset is the extended version of v2.0 of the "Bengali hate speech dataset", which was prepared as a part of our paper titled "Classification Benchmarks for Under-resourced Bengali Language based on Multichannel Convolutional-LSTM Network", in proc. of [7th IEEE International Conference on Data Science and Advanced Analytics (DSAA,2020)](http://dsaa2020.dsaa.co/), October 6-9 2020, Sydney, Australia. This paper also won the [best application paper award at DSAA'2020](https://twitter.com/IEEEDSAA/status/1317599586341462016), which also can be accessed on [arXiv](https://arxiv.org/pdf/2004.07807.pdf) as pre-print.

### Warning!
The data and lexicons contain contenst that are racist, sexist, homophobic, and offensive in many different ways. Datasets are collected and subsequently annotated only for research related purposes. Besides, authors don't take any liability if some statements contain very offensive and hatred statements that are either directed towards a specific person or entity, or generalized towards a group. Please use it with your risk. 

### Data collection and annotation
Bengali articles were collected from numerous sources from Bangladesh and India including a Bengali Wikipedia dump, Bengali news articles (Daily Prothom Alo, Daily Jugontor, Daily Nayadiganta, Anandabazar Patrika, Dainik Jugasankha, BBC, and Deutsche Welle), news dumps of TV channels (NTV, ETV Bangla, ZEE News), books, blogs, sports portal, and social media (Twitter, Facebook pages and groups, LinkedIn). Facebook pages (e.g., celebrities, athletes, sports, and politicians) and newspaper sources were scrutinized because composedly, they have about 50 million followers, and many opinions, hate speech and review texts come or spread out from there. Altogether, our raw text corpus consists of 250 million articles.  

We extend the Bengali Hate Speech Dataset [2] with additional 5,000 labelled memes, making it the largest and only multimodal hate speech dataset in Bengali. We follow a bootstrap approach for data collection, where specific types of texts containing common slurs and terms, either directed towards a specific person or entity or
generalized towards a group, are only considered. Texts were collected from Facebook, YouTube comments, and newspapers. While the “Bengali Hate Speech Dataset” categorized observations into political, personal, geopolitical, religious, and gender abusive hates, we categorized them into hateful and non-hateful, keeping their respective contexts intact. Sample distribution and definition of different types of hates are outlined in table 2. Three annotators (a linguist, a native Bengali speaker, and an NLP researcher) participated in the annotation process. Further, to reduce possible bias, unbiased contents are supplied to the annotators and each label was assigned based on a majority voting on the annotator’s independent opinions. To evaluate the quality of the annotations and to ensure the decision based on the criteria of the objective, we measure inter-annotator agreement w.r.t [Cohen's Kappa statistics](https://en.wikipedia.org/wiki/Cohen%27s_kappa).

### Statistics and frequent words
Following figure shows the most frequently used terms that express different types of hates in Bengali: 

<p align="center"><img src="images/word_cloud_hate.png?" width="400" height="350"></p>

The dataset has 4,500 samples, which has the following distribution: 

<p align="center"><img src="images/stat.jpg?" width="300" height="100"></p>

THe hate column signifies samples that are hatred towards following contexts: 

| Personal Hate | Political Hate |  Religious Hate | Geopoitical Hate | Gender abusive |
| --------------------------| --------------------------| -------------| --------------------------| --------------------------| 
| Directed towards a specific person | Directed towards a political group or person | Directed towards a specific religion | Directed towards a specific country, continent, or regions| Directed towards a specific gender | 

Examples of Bengali hate speech, either directed or generalized towards a specific person, entity, or a group: 

<p align="left"><img src="images/text_memes.png?" width="900" height="350"></p>

Examples of multimodal memes, where texts and visual information add relevant context for the hate speech detection: 

<p align="left"><img src="images/memes.png?" width="900" height="500"></p>

### Data availability and citation request
We will make the dataset publicly available for research. If you want to use the code of this repository or the dataset in your research, please consider citing  folowing paper:

    @inproceedings{karim2022multimodalhate,
      title={Multimodal Hate Speech Detection from Bengali Memes and Texts},
      author={Anonymized for review},
      booktitle={SPELLL},
      pages={1--18},
      year={2022},
      organization={SPELLL}
    }
	
### Contributing
In future, we'll provide an email address, in case readers have any questions.