##Importing Libraries
import pandas as pd
from sodapy import Socrata

##sodapy is a python client for the Socrata Open Data API.
def get_data_from_api():
    try:
    
        client = Socrata("data.buffalony.gov",
                        "9PGAWZ05aDjjp33xR5XagYs9E",
                        username="mancysax@buffalo.edu",
                        password="pass@123")
        results = client.get_all("d6g9-xbgu",
                                select = "case_number,incident_datetime,incident_type_primary,parent_incident_type,hour_of_day,day_of_week,address_1,neighborhood_1,latitude,longitude"
                                )

        # Convert to pandas DataFrame
        results_df = pd.DataFrame.from_records(results)

        results_df.to_csv("CI.csv", sep=',')
        return True
    except Exception as e:
        print(e)