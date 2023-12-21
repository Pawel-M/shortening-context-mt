# Extended results on the Large Contrastive dataset
Here we show the detailed results on the Large Contrastive datasets for the models trained on the IWSLT 2017 en-de dataset.

## Transformer
### context size: 0

```
statistics by ante distance
0 : 4541 5986 0.7586034079518877 
1 : 3459 4566 0.7575558475689882 
2 : 1254 1629 0.7697974217311234 
3 : 675 880 0.7670454545454546 
>3 : 700 939 0.7454739084132055 
```

## Single-encoder

### context size: 1
```
statistics by ante distance
0 : 4602 5986 0.7687938523220849
1 : 3523 4566 0.7715724923346474
2 : 1277 1629 0.7839165131982812
3 : 694 880 0.7886363636363637
>3 : 722 939 0.7689030883919062
```

### context size: 2
```
statistics by ante distance
0 : 4724 5986 0.7891747410624791
1 : 3593 4566 0.7869031975470872
2 : 1306 1629 0.8017188459177409
3 : 695 880 0.7897727272727273
>3 : 739 939 0.7870074547390842
```

### context size: 3
```
statistics by ante distance
0 : 4811 5986 0.8037086535248914
1 : 3698 4566 0.8098992553657468
2 : 1332 1629 0.8176795580110497
3 : 719 880 0.8170454545454545
>3 : 762 939 0.8115015974440895
```

## Multi-encoder

### context size: 1
```
statistics by ante distance
0 : 4532 5986 0.7570998997661209
1 : 3428 4566 0.750766535260622
2 : 1253 1629 0.7691835481890731
3 : 677 880 0.7693181818181818
>3 : 711 939 0.7571884984025559
```

### context size: 2
```
statistics by ante distance
0 : 4611 5986 0.7702973605078517
1 : 3523 4566 0.7715724923346474
2 : 1272 1629 0.7808471454880295
3 : 691 880 0.7852272727272728
>3 : 715 939 0.7614483493077743
```
### context size: 3
```
statistics by ante distance
0 : 4494 5986 0.7507517540928834
1 : 3475 4566 0.761060008760403
2 : 1263 1629 0.7753222836095764
3 : 680 880 0.7727272727272727
>3 : 695 939 0.7401490947816827
```

## Caching Tokens

### context size: 1
```
statistics by ante distance
0 : 4777 5986 0.7980287337119947
1 : 3612 4566 0.7910643889618922
2 : 1315 1629 0.807243707796194
3 : 710 880 0.8068181818181818
>3 : 740 939 0.7880724174653887
```

### context size: 2
```
statistics by ante distance
0 : 4775 5986 0.7976946207818243
1 : 3673 4566 0.8044240035041612
2 : 1316 1629 0.8078575813382444
3 : 715 880 0.8125
>3 : 740 939 0.7880724174653887
```
### context size: 3
```
statistics by ante distance
0 : 4745 5986 0.7926829268292683
1 : 3665 4566 0.8026719229084538
2 : 1324 1629 0.8127685696746471
3 : 707 880 0.803409090909091
>3 : 745 939 0.7933972310969116
```

## Caching Sentence

### context size: 1
```
statistics by ante distance
0 : 4598 5986 0.7681256264617441
1 : 3524 4566 0.7717915024091109
2 : 1274 1629 0.7820748925721301
3 : 706 880 0.8022727272727272
>3 : 724 939 0.7710330138445154
```
### context size: 2
```
statistics by ante distance
0 : 4533 5986 0.7572669562312061
1 : 3494 4566 0.765221200175208
2 : 1254 1629 0.7697974217311234
3 : 692 880 0.7863636363636364
>3 : 702 939 0.7476038338658147
```
### context size: 3
```
statistics by ante distance
0 : 4490 5986 0.7500835282325427
1 : 3464 4566 0.7586508979413052
2 : 1275 1629 0.7826887661141805
3 : 666 880 0.7568181818181818
>3 : 704 939 0.7497337593184239
```

## Shortening - Max Pooling

### context size: 1
```
statistics by ante distance
0 : 4818 5986 0.8048780487804879
1 : 3670 4566 0.803766973280771
2 : 1305 1629 0.8011049723756906
3 : 705 880 0.8011363636363636
>3 : 769 939 0.8189563365282215
```
### context size: 2
```
statistics by ante distance
0 : 4804 5986 0.8025392582692951
1 : 3686 4566 0.8072711344721857
2 : 1322 1629 0.8115408225905464
3 : 710 880 0.8068181818181818
>3 : 761 939 0.8104366347177849
```
### context size: 3
```
statistics by ante distance
0 : 4728 5986 0.78984296692282
1 : 3665 4566 0.8026719229084538
2 : 1324 1629 0.8127685696746471
3 : 711 880 0.8079545454545455
>3 : 732 939 0.7795527156549521
```

## Shortening - Avg Pooling

### context size: 1
```
statistics by ante distance
0 : 4631 5986 0.7736384898095556
1 : 3548 4566 0.777047744196233
2 : 1290 1629 0.7918968692449355
3 : 683 880 0.7761363636363636
>3 : 733 939 0.7806176783812566
```
### context size: 2
```
statistics by ante distance
0 : 4785 5986 0.7993651854326762
1 : 3653 4566 0.8000438020148927
2 : 1316 1629 0.8078575813382444
3 : 719 880 0.8170454545454545
>3 : 749 939 0.7976570820021299
```
### context size: 3
```
statistics by ante distance
0 : 4785 5986 0.7993651854326762
1 : 3688 4566 0.8077091546211126
2 : 1330 1629 0.8164518109269491
3 : 713 880 0.8102272727272727
>3 : 742 939 0.7902023429179978
```

## Shortening - Linear Pooling

### context size: 1
```
statistics by ante distance
0 : 4781 5986 0.7986969595723354
1 : 3669 4566 0.8035479632063075
2 : 1343 1629 0.8244321669736034
3 : 709 880 0.8056818181818182
>3 : 764 939 0.8136315228966986
```
### context size: 2
```
statistics by ante distance
0 : 4717 5986 0.7880053458068828
1 : 3610 4566 0.7906263688129654
2 : 1315 1629 0.807243707796194
3 : 710 880 0.8068181818181818
>3 : 760 939 0.8093716719914803
```
### context size: 3
```
statistics by ante distance
0 : 4731 5986 0.7903441363180755
1 : 3657 4566 0.8009198423127464
2 : 1317 1629 0.8084714548802947
3 : 706 880 0.8022727272727272
>3 : 738 939 0.7859424920127795 
```

## Shortening - Grouping

### context size: 1
```
statistics by ante distance
0 : 4746 5986 0.7928499832943535
1 : 3671 4566 0.8039859833552343
2 : 1327 1629 0.8146101903007981
3 : 703 880 0.7988636363636363
>3 : 741 939 0.7891373801916933
```
### context size: 2
```
statistics by ante distance
0 : 4843 5986 0.8090544604076177
1 : 3712 4566 0.8129653964082347
2 : 1330 1629 0.8164518109269491
3 : 716 880 0.8136363636363636
>3 : 757 939 0.8061767838125665
```
### context size: 3
```
statistics by ante distance
0 : 4733 5986 0.7906782492482459
1 : 3662 4566 0.8020148926850635
2 : 1285 1629 0.7888275015346838
3 : 714 880 0.8113636363636364
>3 : 741 939 0.7891373801916933
```
-----------------------------------

## Shortening - Selecting

### context size: 1
```
statistics by ante distance
0 : 4807 5986 0.8030404276645506
1 : 3700 4566 0.8103372755146737
2 : 1334 1629 0.8189073050951504
3 : 728 880 0.8272727272727273
>3 : 755 939 0.8040468583599574
```
### context size: 2
```
statistics by ante distance
0 : 4799 5986 0.801703975943869
1 : 3668 4566 0.803328953131844
2 : 1326 1629 0.8139963167587477
3 : 713 880 0.8102272727272727
>3 : 739 939 0.7870074547390842
```
### context size: 3
```
statistics by ante distance
0 : 4746 5986 0.7928499832943535
1 : 3668 4566 0.803328953131844
2 : 1334 1629 0.8189073050951504
3 : 700 880 0.7954545454545454
>3 : 764 939 0.8136315228966986
```