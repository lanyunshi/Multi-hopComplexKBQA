# **Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases**
This is the code for the paper:
**[Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases](https://www.aclweb.org/anthology/2020.acl-main.91.pdf)**\
Yunshi Lan, Jing Jiang\
[ACL 2020](https://acl2020.org/) .

If you find this code useful in your research, please cite
>@inproceedings{lan:acl2020,\
>title={Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases},\
>author={Lan, Yunshi and Jiang, Jing},\
>booktitle={Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)},\
>year={2020}\
>}

## **Setups**
All codes were developed and tested in the following environment.
- Ubuntu 16.04
- Python 3.7.1
- Pytorch 1.3.0

Download the code and data:
```
git clone https://github.com/lanyunshi/Multi-hopComplexKBQA.git
pip install requirements.txt
```

## **Download Pre-processed Data**
We evaluate our methods on [WebQuestionsSP](https://www.microsoft.com/en-us/download/details.aspx?id=52763), [ComplexWebQuestions](https://www.tau-nlp.org/compwebq) and [ComplexQuestions](https://github.com/JunweiBao/MulCQA/tree/ComplexQuestions).

The processed data can be downloaded from [link](https://drive.google.com/drive/folders/1sAOUiFbk2ujfXUityIq51p14j9E4HCZ8?usp=sharing) (Please download our latest version. We uploaded another version of pre-processed data on CWQ. Update on 17th July). Please unzip **(in linux system)** and put the folders under the path *data/*.
There are folders for the splits of each dataset, which denote as *splitname_datasetname*. Each folder contains:

- **q.txt**: The file of questions.
- **te.txt**: The file of topic entities (Obtained via NER tool for named entity detection and [Google Knowledge Graph Search API](https://developers.google.com/knowledge-graph)).
- **con.txt**: The file of detected constraints, which usually cannot be detected as topic entities (Obtained via a simple dictionary constructed on training data).
- **a.txt**: The file of answers.
- **g.txt**: The file of ground truth query graphs (No such file is provided for ComplexQuestions).

## **Save Freebase dump in Your Machine**
As we are querying Freebase dynamically, you need install a database engine in your machine. We are using [Virtuoso Open-Source](https://github.com/openlink/virtuoso-opensource). You can follow the instruction to install.
Once you installed the database engine, you can download the raw freebase dump from [link](https://developers.google.com/freebase) and setup your database.
The dump used in our experiments has been pre-processed by removing non-English triplets (See code in *code/FreebaseTool/*).
You can use the code to clean up the raw Freebase dump, which could speed up the query time but may not influence the results.
After the database is installed, you need to replace the *SPARQLPATH* in *code/SPARQL_test.py* file with *your/path/to/database*.
To do a quick test, you can run:
```
python code/SPARQL_test.py
```
The output is as follows (The output maybe different if you didn't do any pre-processing step to the raw dump.):
```
{'head': {'link': [], 'vars': ['name3']}, 'results': {'distinct': False, 'ordered': True, 'bindings': [{'name3': {'type': 'literal', 'xml:lang': 'en', 'value': 'Shelly Wright'}}, {'name3': {'type': 'literal', 'xml:lang': 'en', 'value': 'Jeffrey Probst'}}, {'name3': {'type': 'literal', 'xml:lang': 'en', 'value': 'Lisa Ann Russell'}}]}}
```

If you fail to save Freebase dump in your machine, you **cannot train** a new model using our code **except WBQ (See following section)**.
But you can **still test** our pre-trained models.
The caches of tested datasets are stored in *data/datasetname*. Each folder contains:

- **kb_cache.json**: A dictionary of the cache of searched query graphs *{query: answer}*.
- **m2n_cache.json**: A dictionary of the cache of searched mid *{mid: surface name}*.
- **query_cache.json**: A set of the cache of search query *set(query)*.

## **Download Pre-trained Model**
You can download our pre-trained models from the [link](https://drive.google.com/drive/folders/1Kw1kNXR6IaFTmjLDoPcb_YcwWt8NTc8t?usp=sharing) and put the folders under the path *trained_model/*.

## **Test the Pre-trained Model**
To test our pre-trained model, simply run shell files:
```
./[CWQ|WBQ|CQ]_Runner.sh
```
The predicted query graphs are saved in *trained_model/Best_predcp.txt*. Following the official evaluation matrix of WebQuestionsSP, we measure [hit@1|accuracy|precision|recall|f1] based on predicted query graph.
You can simply run:
```
python code/Evaluation.py \
    --data_path trained_model/[CWQ|WBQ|CQ] \
    --data_file Best \
    --mode eval \
```
If there is no any performance loss due to the absence of the database engine, we can obtain results as follows:

|Dataset|CWQ|WBQ|CQ|
|---|---|---|---|
|Hit@1|39.3|73.3|41.3|
|Accuracy|30.0|66.1|34.0|
|Precision|37.8|73.6|42.0|
|Recall|55.2|80.5|56.6|
|F1|40.4|73.8|43.3|

Please note the hit@1 is calculated based on the accumulated scores of the predicted answers. If there are multiple answers with same scores, we randomly sample one from them.

For official hit@1 of ComplexWebQuestions reported in the paper, instead of considering the top1 ranked query graph, we save top2 ranked query graphs and extract answers based on the accumulated scores. Then we transform the mid answer to surface name answers and send to the author for evaluation following the instruction in their [Webpage](https://www.tau-nlp.org/compwebq).

- Add command ``*--top_k 2*'' to *./CWQ_Runner.sh* and re-run *./CWQ_Runner.sh*
- Run code to re-format the answers:
```
python code/Evaluation.py \
    --data_path trained_model/CWQ \
    --data_file Best \
    --mode trans
```

## **Train a New Model**
If you want to train your model, for example CQ, you can input
```
python code/KBQA_Runner.py  \
        --train_folder  data/train_CQ \
        --dev_folder data/dev_CQ \
        --test_folder data/test_CQ \
        --KB_file data/CQ/kb_cache.json \
        --M2N_file data/CQ/m2n_cache.json \
        --QUERY_file data/CQ/query_cache.json \
        --vocab_file data/CQ/vocab.txt \
        --output_dir trained_model/CQ \
        --config config/bert_config_CQ.json \
        --gpu_id 1\
        --save_model my_model \
        --max_hop_num 2 \
        --num_train_epochs 100 \
        --do_train 1\
        --do_eval 1\
        --do_policy_gradient 1\
        --learning_rate 1e-5 \
        --train_limit_number 150 \
```

It takes long time to train the model at the first beginning as random exploration is conducting, especially on CWQ. With the convergence of the model, the query time will decrease and the training processing becomes faster.

## **Cannot install Database ?**
Don't worry! We provide an implementation on WBQ dataset without installation of the database by uploading the entire cache (in the latest [link](https://drive.google.com/drive/folders/1sAOUiFbk2ujfXUityIq51p14j9E4HCZ8?usp=sharing)) on WBQ folder. You can simply run
```
python code/KBQA_Runner.py  \
        --train_folder  data/train_WBQ \
        --dev_folder data/dev_WBQ \
        --test_folder data/test_WBQ \
        --vocab_file data/WBQ/vocab.txt \
        --KB_file data/WBQ/kb_cache.json \
        --M2N_file data/WBQ/m2n_cache.json \
        --QUERY_file data/WBQ/query_cache.json \
        --output_dir trained_model/WBQ \
        --config config/bert_config.json \
        --gpu_id 1\
        --save_model my_model \
        --max_hop_num 2 \
        --num_train_epochs 100 \
        --do_train 1 \
        --do_eval 1\
        --do_policy_gradient 1 \
        --train_limit_number 150 \
```

The training procedure on WBQ dataset are displayed below:

<img src="https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/figure/WBQ_f1.png" width="300"><img src="https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/figure/WBQ_loss.png" width="300">

## **Too Slow to Train on CWQ with One Stage ?**
To train a model on CWQ dataset fast, you can first pre-train a model with max hop number as 1.
```
python code/KBQA_Runner.py  \
        --train_folder  data/train_CWQ \
        --dev_folder data/dev_CWQ \
        --test_folder data/test_CWQ \
        --vocab_file data/CWQ/vocab.txt \
        --KB_file data/CWQ/kb_cache.json \
        --M2N_file data/CWQ/m2n_cache.json \
        --QUERY_file data/CWQ/query_cache.json \
        --output_dir trained_model/CWQ \
        --config config/bert_config.json \
        --gpu_id 1\
        --load_model trained_model/CWQ/new \
        --save_model my_model_1hop \
        --max_hop_num 1 \
        --num_train_epochs 20 \
        --do_train 1\
        --do_eval 1\
        --do_policy_gradient 1\
        --learning_rate 1e-5 \
```
Next, load the pre-trained model and train with max hop number as 2.
```
python code/KBQA_Runner.py  \
        --train_folder  data/train_CWQ \
        --dev_folder data/dev_CWQ \
        --test_folder data/test_CWQ \
        --vocab_file data/CWQ/vocab.txt \
        --KB_file data/CWQ/kb_cache.json \
        --M2N_file data/CWQ/m2n_cache.json \
        --QUERY_file data/CWQ/query_cache.json \
        --output_dir trained_model/CWQ \
        --config config/bert_config.json \
        --gpu_id 1\
        --load_model trained_model/CWQ/my_model_1hop \
        --save_model my_model_2hop \
        --max_hop_num 2 \
        --num_train_epochs 100 \
        --do_train 1\
        --do_eval 1\
        --do_policy_gradient 1\
        --learning_rate 1e-5 \
```
Finally, load the pre-trained model and train with max hop number as 3.
```
python code/KBQA_Runner.py  \
        --train_folder  data/train_CWQ \
        --dev_folder data/dev_CWQ \
        --test_folder data/test_CWQ \
        --vocab_file data/CWQ/vocab.txt \
        --KB_file data/CWQ/kb_cache.json \
        --M2N_file data/CWQ/m2n_cache.json \
        --QUERY_file data/CWQ/query_cache.json \
        --output_dir trained_model/CWQ \
        --config config/bert_config.json \
        --gpu_id 1\
        --load_model trained_model/CWQ/my_model_2hop \
        --save_model my_model_3hop \
        --max_hop_num 3 \
        --num_train_epochs 15 \
        --do_train 1\
        --do_eval 1\
        --do_policy_gradient 1\
        --learning_rate 1e-5 \
```

The training procedures on CWQ dataset in above 3 stages are displayed below. :

<img src="https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/figure/CWQ_1hop.png" width="300"><img src="https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/figure/CWQ_2hop.png" width="300"><img src="https://github.com/lanyunshi/Multi-hopComplexKBQA/blob/master/figure/CWQ_3hop.png" width="300">

**I found some issues saying that it is too slow to train the model on CWQ dataset, I acknowledge that and it is indeed a problem for me when I train the model for the first time. I updated my collected query graph cache (around 20G) in the [link](https://drive.google.com/file/d/1XpvkSUN0SlBfhnscOwn0T3DU63D2GN74/view?usp=sharing), you can get access to it after you send me a request. Hopefully this will save your training time**


If you have any questions, please open an issue or send an email to yslan.2015@phdcs.smu.edu.sg
