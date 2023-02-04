# Buffalo-Crime
Predictive and non-predictive analysis of 14 years of crime in Buffalo, NY.

# OVERVIEW
In this project, I have used the [CitiStat Buffalo dataset](https://data.buffalony.gov/Public-Safety/Crime-Incidents/d6g9-xbgu) to vizualize crime trends and patterns in the city of Buffalo, New York. 

# OBJECTIVE
With the help of this dataset, we hope to answer and vizualize the following questions: 
1. What is the most common type of crime in Buffalo?
2. What are the most dangerous neighborhoods in Buffalo?
3. Is there a relation between the neighborhood and type of crime in Buffalo?
4. What is the difference in violent crime statistics between the weekdays and the weekends?
5. Is there a relationship between violent crime statistics and seasons?
6. What has been the overall trend of violent crime in Buffalo in the last 14 years?

# WORKFLOW
In this section, let's define a workflow to understand how to use the dataset to answer our questions. Here's some of the steps we'll follow to make sure we get the most accurate answers to our questions:
1. **Collect the latest version of our data.** To do this, we'll use the Socrata Open Data API (SODA) and pull the data using a Python script.
2. **Clean, and normalize our data and systematically store our data.** This is achieved by splitting our original dataset into three tables. The objective of creating these tables is to normalize the data and prevent redundancy and get more accurate results. It also logically organizes our data so that we may view it through different perspectives if required.
3. **Store the data in a database.** The data, once split into the tables, is stored in a database using SQLite module in Python. Once our tables have been created, we can query them using SQL.
4. **Retrive the data contained in the tables to answer our questions.** We can use SQL to write queries to get the data required to answer our questions. Once we have the information we need, we can proceed to creating plots and vizualizations for our data.
5. **Create vizualizations using matplotlib and seaborn.** Finally, we use matplotlib and seaborn to create vizualizations for the data we have retrieved in the previous step. Depending on the question we would like to answer, we use a combination of bar graphs, pie charts and line graphs to vizualise the answers to our questions.
