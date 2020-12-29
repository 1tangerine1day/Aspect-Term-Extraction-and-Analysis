# Aspect-based Sentiment Analysis
## model:
* Bert for Aspect Term Extraction:
* Bert for Aspect-based Sentiment Analysis:

## dataset:
* SemEval-2014 task4:
    * Laptops: 
        * train: 2327
        * test: 636
    * Restaurants:
        * train: 3602
        * test: 1119
    * twitter:
        * train: 6247
        * test: 691
    
<br>![](https://i.imgur.com/KlNHGPo.png)

## performance:
* Aspect Term Extraction:
```
0: unrelated
1: start of aspect term
2: mark of aspect term

Wall time: 23.1 s
              precision    recall  f1-score   support

           0       0.99      0.99      0.99    140373
           1       0.84      0.92      0.88      6486
           2       0.93      0.73      0.82      3837

    accuracy                           0.98    150696
   macro avg       0.92      0.88      0.90    150696
weighted avg       0.99      0.98      0.98    150696
```

* Aspect-based Sentiment Analysis
```
0: negative
1: neutral
2: postive

Wall time: 10.1 s
              precision    recall  f1-score   support

           0       0.72      0.75      0.74       497
           1       0.67      0.74      0.70       710
           2       0.89      0.83      0.86      1239

    accuracy                           0.79      2446
   macro avg       0.76      0.77      0.77      2446
weighted avg       0.79      0.79      0.79      2446
```


## test case:
* For the price you pay this product is very good. However, battery life is a little lack-luster coming from a MacBook Pro.
```
tokens: ['for', 'the', 'price', 'you', 'pay', 'this', 'product', 'is', 'very', 'good', '.', 'however', ',', 'battery', 'life', 'is', 'a', 'little', 'lack', '-', 'lust', '##er', 'coming', 'from', 'a', 'mac', '##book', 'pro', '.']
ATE: ['price', 'battery life']
term: ['price'] class: [2] ABSA: [-2.585547924041748, -1.6089690923690796, 3.54140567779541]
term: ['battery life'] class: [0] ABSA: [5.975338459014893, -2.6804981231689453, -2.68221116065979]
```
* I think Apple is better than Microsoft.
```
tokens: ['i', 'think', 'apple', 'is', 'better', 'than', 'microsoft', '.']
ATE: ['apple', 'microsoft']
term: ['apple'] class: [1] ABSA: [-1.8246030807495117, 2.0324230194091797, 0.0777517780661583]
term: ['microsoft'] class: [0] ABSA: [2.3918569087982178, 0.8508685231208801, -2.396061897277832]
```

* Cyberpunk 2077 freezes constantly, frame rates are terrible, and it's extremely frustrating to try to play.
```
tokens: ['cyber', '##pu', '##nk', '207', '##7', 'freeze', '##s', 'constantly', ',', 'frame', 'rates', 'are', 'terrible', ',', 'and', 'it', "'", 's', 'extremely', 'frustrating', 'to', 'try', 'to', 'play', '.']
ATE: ['cyberpu', 'frame rates']
term: ['cyberpu'] class: [0] ABSA: [4.44415283203125, -0.36560752987861633, -3.3459084033966064]
term: ['frame rates'] class: [0] ABSA: [5.2562408447265625, -2.305537700653076, -2.0652124881744385]
```
