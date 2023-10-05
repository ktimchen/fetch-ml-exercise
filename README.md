<p align="center">
  <img height="300" src="https://github.com/ktimchen/fetch-ml-exercise/assets/36734709/790ce6ac-f3e2-4259-841d-625f35825d39">
</p>


Given a number of months (e.g. 6), this app returns the approximate number of scanned receipts for each month after 2021 (e.g. from Jan 2022 to Jun 2022).

### Build and run application
Build the Docker image manually by cloning the Git repo
```
$ git clone https://github.com/ktimchen/fetch-ml-exercise.git
$ cd fetch-ml-exercise
$ docker build -t fetch-exercise .
```
Create a container from the image
```
$ docker run -it -p 7001:5000 --rm fetch-exercise 
```

Now head over to http://localhost:7001, type in the forecast length (in months), press enter (or click the Forecast button)
and enjoy the predictions! 


### Assumptions / Details
- We are asked to develop an algorithm to predict the number of scanned receipts _**for each month**_ of 2022.
Hence, I am forecasting on the monthly level.
- I've decided to use a simple linear regression because:
  - Only 2021 numbers are available
  - I don't have enough domain knowledge/data to make assumptions, e.g. what does the 2020 data look like? Are August numbers always higher than June numbers? Is there a monthly seasonality?    
- There is a slight dip in February 2021 - but we don't have enough data to make a judgment. Did the same happen in 2020? Is this an outlier? Was the app broken in that month?
- The MAPE (mean absolute percentage error) on the test set of the last three months is ~1% which is fantastic! Real world data is rarely this predictable.
To compute MAPE, run 
    ```
    python3 time_series_model.py 
    ```

#
#### TO-DOS
- add unit tests (make sure `clean_receipts` does what it's supposed to do, check that `TSModel` has correct `.fit()` and `.predict()` methods)
- add typing everywhere
- implement more sophisticated models
- dockerize the accuracy computation
- ...
