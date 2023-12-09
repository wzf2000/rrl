## Data set description
The description of datasets used for regression task are listed in the table below:

|Dataset (Links) | \#Instances	| \#features	| feature type|
| --- | --- | --- | --- |
|[BostonHousing](https://www.kaggle.com/datasets/altavish/boston-housing-dataset)| 506 | 13 | mixed|
|[OnlineNewsPopularity](https://archive.ics.uci.edu/dataset/186/wine+quality)| 39644 | 59 | mixed|
|[WineQuality](https://archive.ics.uci.edu/dataset/332/online+news+popularity)| 6497 | 11 | mixed|

For more information about these data sets, you can click the corresponding links.


## Data set format
Each data set corresponds to two files, `*.data` and `*.info`. The `*.data` file stores the data for each instance. The `*.info` file stores the information for each feature.

You can download processed data from: https://drive.google.com/drive/folders/1OvPjTgzgF7n9Ku7aczo6KfQJVaedkbRE?usp=drive_link

#### *.data
One row in `*.data` corresponds to one instance and one column corresponds to one feature (including the predictive variable).

For example, the `BostonHousing.data`:

|  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00632 | 18.0 | 2.31 | 0.0 | 0.538 | 6.575 | 65.2 | 4.0900 | 1 | 296 | 15.3 | 396.90 | 4.98 | 24.0 |
| 0.02731 | 0.0 | 7.07 | 0.0 | 0.469 | 6.421 | 78.9 | 4.9671 | 2 | 242 | 17.8 | 396.90 | 9.14 | 21.6 |
| ......|

#### *.info
One row (except the last row) in `*.info` corresponds to one feature (including the predictive variable). The order of these features must be the same as the columns in `*.data`. The first column is the feature name, and the second column indicates the characteristics of the feature, i.e., continuous or discrete. 
The last row does not correspond to one feature. It specifies the position of the predictive variable column.

For example, the `BostonHousing.info`:

| | |
| --- | --- |
|CRIM | continuous |
|ZN | continuous |
|INDUS | continuous |
|CHAS | discrete |
|NOX | continuous |
|RM | continuous |
|AGE | continuous |
|DIS | continuous |
|RAD | discrete |
|TAX | continuous |
|PTRATIO | continuous |
|B | continuous |
|LSTAT | continuous |
|MEDV | continuous |
|LABEL_POS | -1 |
