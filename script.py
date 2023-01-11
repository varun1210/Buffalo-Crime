import pandas as pd
import sqlite3
from sqlite3 import Error
import numpy as np
import matplotlib.pyplot as plt
import get_csv_api
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

csv_file = "CI.csv"
db_file = "crime_database.db"

column_mapper_dict = {
    "case_number" : "Case Number",
    "incident_datetime" : "Incident Datetime",
    "incident_type_primary" : "Incident Type Primary",
    "parent_incident_type" : "Parent Incident Type",
    "hour_of_day" : "Hour of Day", 
    "day_of_week" : "Day of Week",
    "address_1" : "Address",
    "neighborhood_1" : "Neighborhood",
    "latitude" : "Latitude",
    "longitude" : "Longitude"
}


def create_connection(db_file, delete_db=False):
    import os
    if delete_db and os.path.exists(db_file):
        os.remove(db_file)

    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
    except Error as e:
        print(e)

    return conn


def create_table(conn, create_table_sql, drop_table_name=None):
    
    if drop_table_name:
        try:
            c = conn.cursor()
            c.execute("""DROP TABLE IF EXISTS %s""" % (drop_table_name))
        except Error as e:
            print(e)
    
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
        
def execute_sql_statement(sql_statement, conn):
    cur = conn.cursor()
    cur.execute(sql_statement)

    rows = cur.fetchall()

    return rows

def insert_product_table(conn, values):
    sql = ''' INSERT OR IGNORE INTO varun_table(fname, lname) values(?, ?) '''
    cur = conn.cursor()
    cur.executemany(sql, values)
    return cur.lastrowid


def create_incident_type(conn, csv_file):
    create_sql = """
    CREATE TABLE IF NOT EXISTS Incident_Type (
        Incident_Type_Primary TEXT NOT NULL PRIMARY KEY,
        Parent_Incident_Type TEXT
    );
    """
    insert_sql = """INSERT OR IGNORE INTO Incident_Type(Incident_Type_Primary, Parent_Incident_Type) values(?, ?)"""
    try:
        create_table(create_table_sql=create_sql, conn=conn, drop_table_name="Incident_Type")
        data = pd.read_csv(csv_file)
        data = data[["Incident Type Primary", "Parent Incident Type"]]
        data = list(data.itertuples(index=False, name=None))
        cur = conn.cursor()
        cur.executemany(insert_sql, data)
        return cur.lastrowid
    except Error as e:
        print(e)

def create_incident_info(conn, csv_file):
    create_sql = """
    CREATE TABLE IF NOT EXISTS Incident_Info (
        Case_Number TEXT NOT NULL PRIMARY KEY,
        Incident_Datetime TEXT,
        Incident_Type_Primary TEXT,
        Hour_of_Day INTEGER, 
        Day_of_Week TEXT
    );
    """
    insert_sql = """INSERT OR IGNORE INTO Incident_Info(Case_Number, Incident_Datetime, Incident_Type_Primary, Hour_of_Day, Day_of_Week) values(?, ?, ?, ?, ?)"""
    try:
        create_table(create_table_sql=create_sql, conn=conn, drop_table_name="Incident_Info")
        data = pd.read_csv(csv_file)
        data = data[["Case Number", "Incident Datetime", "Incident Type Primary", "Hour of Day", "Day of Week"]]
        data = list(data.itertuples(index=False, name=None))
        cur = conn.cursor()
        cur.executemany(insert_sql, data)
        return cur.lastrowid
    except Error as e:
        print(e)

def create_incident_location_org(conn, csv_file):
    create_sql = """
    CREATE TABLE IF NOT EXISTS Incident_Location (
        Address TEXT,
        Latitude REAL,
        Longitude REAL,
        Neighborhood TEXT,
        Case_Number TEXT,
        FOREIGN KEY(Case_Number) REFERENCES Incident_Info(Case_Number)
    );
    """
    insert_sql = """INSERT OR IGNORE INTO Incident_Location(Address, Latitude, Longitude, Neighborhood, Case_Number) values(?, ?, ?, ?, ?)"""
    try:
        create_table(create_table_sql=create_sql, conn=conn, drop_table_name="Incident_Location")
        data = pd.read_csv(csv_file)
        data = data[["Address", "Latitude", "Longitude", "Neighborhood", "Case Number"]]
        data = list(data.itertuples(index=False, name=None))
        cur = conn.cursor()
        cur.executemany(insert_sql, data)
        return cur.lastrowid
    except Error as e:
        print(e)

def create_incident_location(conn, csv_file):
    create_sql = """
    CREATE TABLE IF NOT EXISTS Incident_Location (
        Address TEXT,
        Latitude REAL,
        Longitude REAL,
        Neighborhood TEXT,
        Case_Number TEXT,
        FOREIGN KEY(Case_Number) REFERENCES Incident_Info(Case_Number)
    );
    """
    insert_sql = """INSERT OR IGNORE INTO Incident_Location(Address, Latitude, Longitude, Neighborhood, Case_Number) values(?, ?, ?, ?, ?)"""

    def trim_lower(input_row, column_name):
        if((not input_row[column_name]) and (type(input_row[column_name]) == str)):
            return input_row[column_name].strip().lower()
        else:
            if not input_row[column_name]:
                return None
            return str(input_row[column_name])

    try:
        create_table(create_table_sql=create_sql, conn=conn, drop_table_name="Incident_Location")
        data = pd.read_csv(csv_file)
        Incident_Location_csv = data[["Address", "Latitude", "Longitude", "Neighborhood", "Case Number"]]
        non_empty_lats = Incident_Location_csv[["Address", "Latitude", "Longitude"]]
        non_empty_lats = non_empty_lats.drop_duplicates()
        non_empty_lats = non_empty_lats.query("`Latitude`.notnull() and `Longitude`.notnull()")
        non_empty_lats = non_empty_lats.groupby(["Address"], as_index=False)[["Latitude", "Longitude"]].mean()
        non_empty_lats["Address_exists"] = non_empty_lats.apply(lambda row: trim_lower(row, "Address"), axis = 1)
        Incident_Location_csv["Address_not_exists"] = Incident_Location_csv.apply(lambda row: trim_lower(row, "Address"), axis = 1)
        Incident_Location_csv = Incident_Location_csv.merge(non_empty_lats, how="left", left_on="Address_not_exists", right_on="Address_exists")
        Incident_Location_csv = Incident_Location_csv.drop_duplicates()
        Incident_Location_csv["Address"] = Incident_Location_csv[['Address_x', 'Address_y']].bfill(axis=1).iloc[:, 0]
        Incident_Location_csv["Latitude"] = Incident_Location_csv[['Latitude_x', 'Latitude_y']].bfill(axis=1).iloc[:, 0]
        Incident_Location_csv["Longitude"] = Incident_Location_csv[['Longitude_x', 'Longitude_y']].bfill(axis=1).iloc[:, 0]
        Incident_Location_csv = Incident_Location_csv[["Address", "Latitude", "Longitude", "Neighborhood", "Case Number"]]
        non_empty_nhoods = Incident_Location_csv[["Address", "Neighborhood"]]
        non_empty_nhoods = non_empty_nhoods.drop_duplicates()
        non_empty_nhoods = non_empty_nhoods.query("`Neighborhood`.notnull()")
        non_empty_nhoods = non_empty_nhoods.groupby(["Address"], as_index=False)[["Neighborhood"]].first()
        non_empty_nhoods["Address_exists"] = non_empty_nhoods.apply(lambda row: trim_lower(row, "Address"), axis = 1)
        Incident_Location_csv["Address_not_exists"] = Incident_Location_csv.apply(lambda row: trim_lower(row, "Address"), axis = 1)
        Incident_Location_csv = Incident_Location_csv.merge(non_empty_nhoods, how="left", left_on="Address_not_exists", right_on="Address_exists")
        Incident_Location_csv = Incident_Location_csv.drop_duplicates()
        Incident_Location_csv["Address"] = Incident_Location_csv[['Address_x', 'Address_y']].bfill(axis=1).iloc[:, 0]
        Incident_Location_csv["Neighborhood"] = Incident_Location_csv[['Neighborhood_x', 'Neighborhood_y']].bfill(axis=1).iloc[:, 0]
        Incident_Location_csv = Incident_Location_csv[["Address", "Latitude", "Longitude", "Neighborhood", "Case Number"]]
        data = list(Incident_Location_csv.itertuples(index=False, name=None))
        cur = conn.cursor()
        cur.executemany(insert_sql, data)
        return cur.lastrowid
    except Error as e:
        print(e)
       

try:
    api_call = get_csv_api.get_data_from_api()
    connection = create_connection(db_file=db_file, delete_db=True)
    data = pd.read_csv(csv_file)
    data.rename(columns = column_mapper_dict, inplace = True)
    data.to_csv("CI.csv", sep=',')
    create_incident_type(conn=connection, csv_file=csv_file)
    create_incident_info(conn=connection, csv_file=csv_file)
    create_incident_location(conn=connection, csv_file=csv_file)
    Incident_Type_csv = pd.read_sql_query("select * from Incident_Type", connection)
    Incident_Info_csv = pd.read_sql_query("select * from Incident_Info", connection)
    Incident_Location_csv = pd.read_sql_query("select * from Incident_Location", connection)
    print(Incident_Type_csv)
    Incident_Type_csv.to_csv("Incident_Type_table.csv")
    print(Incident_Info_csv)
    Incident_Info_csv.to_csv("Incident_Info_table.csv")
    print(Incident_Location_csv)
    Incident_Location_csv.to_csv("Incident_Location_table.csv")

    # TOP CRIMES VIZ.
    top_crimes_query = """
    select upper(substr(t.incident_type_primary, 1, 1)) || lower(substr(t.incident_type_primary, 2, length(t.incident_type_primary))) as Incident_Type_Primary, count(i.case_number) as cases
    from incident_info i 
    left join incident_type t
    on i.incident_type_primary = t.incident_type_primary
    where i.incident_type_primary is not null
    group by t.incident_type_primary
    order by cases desc
    limit 5
    """
    top_crimes_data = pd.read_sql_query(top_crimes_query, connection)
    ax = sns.barplot(data=top_crimes_data, x="cases", y="Incident_Type_Primary", palette="pastel")
    ax.set(xlabel='Cases', ylabel='Incident Type', title="Crime Type in Buffalo")
    plt.savefig("Crime Type in Buffalo.png")
    ax.cla()
    plt.clf()

    # TOP NEIGHBORHOODS VIZ.
    top_neighborhoods_query = """
    select upper(substr(neighborhood, 1, 1)) || lower(substr(neighborhood, 2, length(neighborhood))) as neighborhood, count(case_number) as cases
    from incident_location
    where neighborhood is not null and lower(trim(neighborhood)) <> "unknown"
    group by neighborhood
    order by cases desc
    """
    top_neighborhoods_data = pd.read_sql_query(top_neighborhoods_query, connection)
    fig = plt.figure(figsize=(12, 7))
    ax = sns.barplot(data=top_neighborhoods_data, x="cases", y="neighborhood", palette="pastel")
    ax.set(xlabel='Cases', ylabel='Neighborhood', title="Most Dangerous Neighborhoods in Buffalo")
    plt.savefig("Most Dangerous Neighborhoods in Buffalo.png")
    ax.cla()
    plt.clf()

    # WEEKEND VS WEEKDAY CRIMES
    weekend_weekday_sql = """
    select
    type_of_day,
    case 
    when type_of_day = 'weekend' then cases / 1564
    else cases / 3914 
    end as crime_per_day 
    from 
        (select 
        case
        when trim(lower(day_of_week)) in ('saturday', 'sunday') then 'Weekend'
        else 'Weekday'
        end as type_of_day,
        count(distinct case_number) as cases
        from incident_info
        where day_of_week is not null and case_number is not null
        and cast(substr(incident_datetime, 1, 4) as decimal) between 2007 and 2021
        group by type_of_day
    )
    """
    weekend_weekday_data = pd.read_sql_query(weekend_weekday_sql, connection)
    type_of_day = weekend_weekday_data.type_of_day.values.tolist()
    crime_per_day = weekend_weekday_data.crime_per_day.values.tolist()
    colors = sns.color_palette('pastel')[0:2]
    plt.title("Crime in Buffalo - Weekend vs. Weekday")
    plt.pie(crime_per_day, labels = type_of_day, colors = colors, autopct='%.0f%%')
    plt.savefig("Crime in Buffalo - Weekend vs Weekday")
    ax.cla()
    plt.clf() 

    # VIOLENT CRIMES BY SEASON
    violet_crimes_month_query = """
    select 
    case
    when substr(incident_datetime, 6, 2) in ('03', '04', '05') then 'Spring'
    when substr(incident_datetime, 6, 2) in ('06', '07', '08') then 'Summer'
    when substr(incident_datetime, 6, 2) in ('09', '10', '11') then 'Autumn'
    when substr(incident_datetime, 6, 2) in ('12', '01', '02') then 'Winter'
    end as season, 
    count(case_number) as violent_crimes
    from incident_info
    left join incident_type
    on incident_info.incident_type_primary = incident_type.incident_type_primary
    where lower(trim(incident_info.incident_type_primary)) in ('assault', 'homicide', 'manslaughter', 'murder', 'rape', 'sexual assault', 'sodomy')
    and incident_datetime like "____-__-__%"
    group by season
    order by violent_crimes desc
    """
    violent_crimes_month_data = pd.read_sql_query(violet_crimes_month_query, connection)
    seasons = violent_crimes_month_data.season.values.tolist()
    violent_crimes = violent_crimes_month_data.violent_crimes.values.tolist()
    colors = sns.color_palette('pastel')[0:4]
    exp = [0.2, 0, 0, 0]
    plt.title("Violent Crimes by Season")
    plt.pie(violent_crimes, labels = seasons, colors = colors, autopct='%.0f%%', explode=exp)
    plt.savefig("Violent Crimes by Season")
    ax.cla()
    plt.clf()

    # MONTH WISE CRIME TREND OVER YEARS
    month_wise_trend_query = """
    select
    case
    when substr(incident_datetime, 6, 2) = '01' then 'January'
    when substr(incident_datetime, 6, 2) = '02' then 'February'
    when substr(incident_datetime, 6, 2) = '03' then 'March'
    when substr(incident_datetime, 6, 2) = '04' then 'April'
    when substr(incident_datetime, 6, 2) = '05' then 'May'
    when substr(incident_datetime, 6, 2) = '06' then 'June'
    when substr(incident_datetime, 6, 2) = '07' then 'July'
    when substr(incident_datetime, 6, 2) = '08' then 'August'
    when substr(incident_datetime, 6, 2) = '09' then 'September'
    when substr(incident_datetime, 6, 2) = '10' then 'October'
    when substr(incident_datetime, 6, 2) = '11' then 'November'
    when substr(incident_datetime, 6, 2) = '12' then 'December'
    end as Month,
    substr(incident_datetime, 1, 4) as year,
    count(case_number) as violent_crimes
    from incident_info
    left join incident_type
    on incident_info.incident_type_primary = incident_type.incident_type_primary
    where lower(trim(incident_info.incident_type_primary)) in ('assault', 'homicide', 'manslaughter', 'murder', 'rape', 'sexual assault', 'sodomy')
    and incident_datetime like "____-__-__%"
    and cast(substr(incident_datetime, 1, 4) as decimal) between 2007 and 2021
    group by Month, year
    order by year, Month, violent_crimes desc
    """
    """ in ('2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021')"""
    line_chart_data = pd.read_sql_query(month_wise_trend_query, connection)
    ax = sns.lineplot(data=line_chart_data, x="year", y="violent_crimes", hue="Month", hue_order=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    ax.set(xlabel='Year', ylabel='Violent Crimes', title="16 Year Violent Crime Trends in Buffalo")
    plt.legend(loc="upper right")
    plt.savefig("16 Year Violent Crime Trends in Buffalo")
    ax.cla()
    plt.clf()


    # INCIDENT TYPE AGAINST CASES NUMBER
    sql_heatmap_data = """
    select l.neighborhood, t.parent_incident_type, count(l.case_number) as cases
    from incident_location l
    left join incident_info i
    on l.case_number = i.case_number
    left join incident_type t
    on i.incident_type_primary = t.incident_type_primary
    where trim(lower(l.neighborhood)) in ('broadway fillmore', 'central', 'kensington-bailey', 'north park', 'genesee-moselle')
    and trim(lower(t.parent_incident_type)) in ('theft', 'assault', 'breaking & entering', 'theft of vehicle', 'robbery')
    group by l.neighborhood, t.parent_incident_type
    order by l.neighborhood, t.parent_incident_type, cases desc
    """
    heatmap_data = pd.read_sql_query(sql_heatmap_data, connection)
    neighborhoods = [*set(heatmap_data.Neighborhood.values.tolist())]
    neighborhoods.sort()
    parent_incident_type = [*set(heatmap_data.Parent_Incident_Type.values.tolist())]
    parent_incident_type.sort()
    count = 0
    outerlist = []
    inner_list = []
    for index, rows in heatmap_data.iterrows():
        if(count == 24):
            inner_list.append(int(rows.cases))
            outerlist.append(inner_list)
            break
        if((count % 5 == 0) and (count != 0)):
            outerlist.append(inner_list)
            inner_list = []
        inner_list.append(int(rows.cases))
        count += 1
    array = np.array(outerlist)
    fig, ax = plt.subplots()
    im = ax.imshow(outerlist)
    ax.set_xticks(np.arange(len(neighborhoods)), labels=neighborhoods)
    ax.set_yticks(np.arange(len(parent_incident_type)), labels=parent_incident_type)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    for i in range(len(neighborhoods)):
        for j in range(len(parent_incident_type)):
            text = ax.text(j, i, array[i, j], ha="center", va="center", color="w")
    ax.set_title("Crime Distribution by Neighborhood")
    fig.tight_layout()
    plt.savefig("Crime Distribution by Neighborhood")

    #---- MODEL ----
    model_sql = """
    select 
    case 
    when trim(lower(i.incident_type_primary)) = 'larceny/theft' then 0
    else 1
    end as incident_type_primary,
    i.hour_of_day, i.day_of_week, l.address, l.neighborhood,
    substr(i.incident_datetime, 9, 2) as date,
    substr(i.incident_datetime, 6, 2) as month,
    substr(i.incident_datetime, 1, 4) as year
    from incident_info i
    left join incident_location l 
    on i.case_number = l.case_number
    where l.address is not null and i.incident_type_primary is not null
    and i.incident_datetime like "____-__-__%"
    and l.neighborhood is not null and i.hour_of_day is not null and i.day_of_week is not null
    """
    model_rename_columns = {
        "incident_type_primary" : "Incident Type Primary",
        "Hour_of_Day" : "Hour of Day",
        "Day_of_Week" : "Day of Week"
    }
    model_data = pd.read_sql_query(model_sql, connection)
    model_data.rename(columns = model_rename_columns, inplace = True)
    from sklearn import preprocessing
    # label_encoder object knows how to understand word labels.
    label_encoder = preprocessing.LabelEncoder()
    # Encode labels in column 'species'.
    model_data['Day of Week']= label_encoder.fit_transform(model_data['Day of Week'])
    model_data['Address']= label_encoder.fit_transform(model_data['Address'])
    model_data['Neighborhood']= label_encoder.fit_transform(model_data['Neighborhood'])
    model_data['month']= label_encoder.fit_transform(model_data['month'])
    model_data['year']= label_encoder.fit_transform(model_data['year'])
    model_data['date']= label_encoder.fit_transform(model_data['date'])
    X = model_data.iloc[:,1:7]
    Y = model_data.iloc[:,0:1]
    seed = 42
    test_size = 0.2
    X_train, X_test,y_train,y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
    model = XGBClassifier(max_depth=9, max_leaves=9)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))

    conf_mat = confusion_matrix(y_train, y_pred)
    print(conf_mat)

except Exception as e:
    print(e)