# **Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases**
This is the code for the paper:
**[Query Graph Generation for Answering Multi-hop Complex Questions from Knowledge Bases]
()**\
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
All code was developed and tested in the following environment. 
- Ubuntu 16.04
- Python 3.7.1
- Pytorch 1.1.0

Download the code and data:
```
git clone https://github.com/lanyunshi/Multi-hopComplexKBQA.git
```

## **Save Freebase dump in Your Machine**
As we are querying Freebase dynamically, you can install a database engine in your machine. We are using [Virtuoso Open-Source](https://github.com/openlink/virtuoso-opensource). You can follow the instruction to install.
Once you installed the database engine, you can download our freebase dump from [link]() and save it in your data path.
To do a quick test, you can run:
```
python code/SQL_test.py
```
The output is as follows:
TO DO !!!!!

Please note if you fail to save Freebase dump in your machine, you cannot train a new model using our code.

## **Download
