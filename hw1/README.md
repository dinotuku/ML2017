## Predict future PM2.5 values using linear regression  

###Data description
資料是從中央氣象局下載的真實觀測資料。  
作業中使用豐原站的觀測記錄，分成 train set 跟 test set。

train.csv：每個月前 20 天的完整資料  

test_X.csv：從剩下的10天資料中取樣出連續的10小時為一筆，前九小時的所有觀測數據當作feature，第十小時的PM2.5當作answer。一共取出240筆不重複的test data，請根據feauure預測這240筆的PM2.5。

###Train linear regression model
```
bash hw1.sh [input file 1] [input file 2] [output file]
```

###Use the best model
```
bash hw1_best.sh [input file 1] [input file 2] [output file]
```