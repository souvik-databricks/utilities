# Databricks notebook source
# DBTITLE 1,Essential Imports
import requests as req
import pandas as pd
import json
from pyspark.sql import functions as f
from pyspark.sql import types as t
import base64

# COMMAND ----------

# DBTITLE 1,Getting context info (instance name, token for api call)
context = json.loads(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
)
instancename = context["tags"]["browserHostName"]
token = (
    dbutils.widgets._entry_point.getDbutils().notebook().getContext().apiToken().get()
)

# COMMAND ----------

# DBTITLE 1,Getting all of the jobs
job_lst = []

headers = {
    "Authorization": f"Bearer {token}",
}
query_params = {"limit": 25, "offset": 0, "expand_tasks": "true"}
response = req.get(
    f"https://{instancename}/api/2.1/jobs/list", headers=headers, params=query_params
).json()

has_more = response["has_more"]

job_lst = job_lst + response["jobs"]

while has_more:
    query_params["offset"] = query_params["offset"] + query_params["limit"]
    recurse_response = req.get(
        f"https://{instancename}/api/2.1/jobs/list",
        headers=headers,
        params=query_params,
    ).json()
    job_lst = job_lst + recurse_response["jobs"]
    has_more = recurse_response["has_more"]
print("Job List count:", len(job_lst))

# COMMAND ----------

job_lst

# COMMAND ----------

# DBTITLE 1,Function to extract the notebook path object
def get_notebook_path(dic_obj):
    if "notebook_task" in dic_obj.keys():
        notebook_path = dic_obj["notebook_task"].get("notebook_path")
        source = dic_obj["notebook_task"].get("source")
        return {"notebook_path": notebook_path, "source": source}
    else:
        return {
            "notebook_path": "No notebooks tasks here",
            "source": "No notebooks tasks here",
        }

# COMMAND ----------

# DBTITLE 1,Parsing the jobs information which is pulled from the API
parsed_job_lst = [
    (
        i.get("creator_user_name", "Not available"),
        i["job_id"],
        any([True for data in i["settings"]["tasks"] if "sql_task" in data.keys()]),
        [
            get_notebook_path(data)
            for data in i["settings"]["tasks"]
            if "notebook_task" in data.keys()
        ],
    )
    for i in job_lst
]

# COMMAND ----------

parsed_job_lst

# COMMAND ----------

# DBTITLE 1,Converting our information to spark dataframe
# data_schema = t.StructType([
#   t.StructField('creator_user_name',t.StringType()),
#   t.StructField('job_id',t.LongType()),
#   t.StructField('contains_sql_task',t.BooleanType()),
#   t.StructField('notebook_tasks',t.ArrayType(t.MapType(t.StringType(),t.StringType())))
# ])
data_schema = t.StructType(
    [
        t.StructField("creator_user_name", t.StringType()),
        t.StructField("job_id", t.LongType()),
        t.StructField("contains_sql_task", t.BooleanType()),
        t.StructField(
            "notebook_tasks",
            t.ArrayType(
                t.StructType(
                    [
                        t.StructField("notebook_path", t.StringType()),
                        t.StructField("source", t.StringType()),
                    ]
                )
            ),
        ),
    ]
)
job_data = spark.createDataFrame(parsed_job_lst, data_schema)

# COMMAND ----------

job_data.display()

# COMMAND ----------

# DBTITLE 1,Transformations to flatten our data's nested structure for easy analysis
exploded_job_data = job_data.withColumn(
    "notebook_tasks", f.explode_outer("notebook_tasks")
)
cleaned_job_data = exploded_job_data.select(
    "creator_user_name",
    "job_id",
    "contains_sql_task",
    f.col("notebook_tasks.notebook_path").alias("notebook_path"),
    f.col("notebook_tasks.source").alias("source"),
)

# COMMAND ----------

exploded_job_data.display()
cleaned_job_data.display()

# COMMAND ----------

# DBTITLE 1,Defining function to extract language of the workload  and wrapping it in spark udf for parallelization
@f.udf("string")
def workload_type(notebook_path, source):
    lang = None
    if source == "WORKSPACE":
        auth_header = {
            "Authorization": f"Bearer {token}",
        }
        req_data = {"path": notebook_path}
        get_resp = req.get(
            f"https://{instancename}/api/2.0/workspace/list",
            headers=auth_header,
            params=req_data,
        ).json()
        if "objects" in get_resp.keys():
            lang = get_resp["objects"][0].get("language")
        else:
            lang = "Resource unavailable"
    else:
        lang = "Remote Notebook so language unidentifiable"

    return lang

# COMMAND ----------

# DBTITLE 1,Defining function to extract SQL cells in notebooks and how much percentage
@f.udf(
    t.StructType(
        [
            t.StructField("sql_count", t.IntegerType()),
            t.StructField("sql_percent", t.DoubleType()),
        ]
    )
)
def sql_count_and_percent(notebook_path, source):
    sql_cell_count, sql_percentage = 0, 0
    if source == "WORKSPACE":
        auth_header = {
            "Authorization": f"Bearer {token}",
        }
        data = {"path": notebook_path, "format": "JUPYTER", "direct_download": False}
        get_resp = req.get(
            f"https://{instancename}/api/2.0/workspace/export",
            headers=headers,
            json=data,
        ).json()
        if "content" in get_resp.keys():
            base64_string = get_resp["content"]
            base64_bytes = base64_string.encode("utf-8")
            nb_string_bytes = base64.b64decode(base64_bytes)
            nb_string = nb_string_bytes.decode("utf-8")
            nb_json = json.loads(nb_string)
            for i in nb_json["cells"]:
                if "%sql" in i["source"][0]:
                    sql_cell_count += 1
            sql_percentage = round((sql_cell_count / len(nb_json["cells"]))*100,2)
        else:
            sql_cell_count, sql_percentage = None, None
    else:
        sql_cell_count, sql_percentage = None, None
    return {"sql_count": sql_cell_count, "sql_percent": sql_percentage}

# COMMAND ----------

# DBTITLE 1,Finally our jobs data curated with our required information
final_job_data = cleaned_job_data.withColumn(
    "lang", workload_type("notebook_path", "source")
).withColumn("sql_stats", sql_count_and_percent("notebook_path", "source"))
final_job_data = final_job_data.withColumn(
    "notebook_workload",
    f.struct(
        "notebook_path",
        "source",
        "lang",
        f.col("sql_stats.sql_count"),
        f.col("sql_stats.sql_percent"),
    ),
).drop("notebook_path", "source", "lang", "sql_stats")
final_job_data.display()

# COMMAND ----------

# DBTITLE 1,Filtering the curated data to see which jobs contains SQL workloads (Tier - 1 targets)
final_job_data.filter(
    "notebook_workload.lang = 'SQL' or contains_sql_task = true"
).groupBy("creator_user_name", "job_id", "contains_sql_task").agg(
    f.collect_list("notebook_workload").alias("notebook_workloads")
).display()

# COMMAND ----------

# DBTITLE 1,Getting the workloads which has notebook tasks of Non-SQL languages but contain  a lot of sql cells (Tier-2 targets)
final_job_data.orderBy(
    f.col("notebook_workload.sql_count").desc(),
    f.col("notebook_workload.sql_percent").desc(),
).groupBy("creator_user_name", "job_id", "contains_sql_task").agg(
    f.collect_list("notebook_workload").alias("notebook_workloads")
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC Repos: `file:/Workspace/Repos`
# MAGIC 
# MAGIC DBFS: `file:/dbfs/FileStore`
