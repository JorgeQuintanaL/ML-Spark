import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object DOP {
  def main(args: Array[String]) =
  {
    val sparkSession = SparkSession.builder
      .master("local[*]")
      .appName("DOP")
      .config("spark.sql.parquet.compression.codec", "gzip")
      .getOrCreate()

    import sparkSession.implicits._
    val Data = sparkSession
      .read.option("header", "true")
      .option("charset", "UTF8")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv("./data/Data.csv")

    val CleanData = Data
      .drop("studentid")
      .drop("schoolid")
      .drop("label_school_start_cal_year")
      .drop("label_school_year")
      .drop("feature_school_start_cal_year")
      .drop("feature_school_year")
      .drop("retained_started_sy")
      .drop("retained_completed_sy_inc")
      .drop("retained_started_sy_inc")
      .drop("retained_funding_sy")
      .drop("retained_funding_sy_inc")
      .drop("income_annual")
      .drop("income_grouped")
      .drop("free_reduced_lunch_group_name")
      .drop("n_activities_todo")
      .drop("n_activities_done")
      .drop("avg_ela_course_grade")
      .drop("avg_math_course_grade")
      .drop("avg_science_course_grade")
      .drop("income_frequency")

    val To_Model = CleanData
      .filter($"retained_completed_sy" === "RETAINED" ||
      $"retained_completed_sy" === "IN YEAR WITHDRAW" ||
      $"retained_completed_sy" === "BETWEEN SY WITHDRAW")

    val Model_Flag = To_Model
      .withColumn("Flag", when($"retained_completed_sy" === "RETAINED", 1.0).otherwise(0))
      .drop("retained_completed_sy")
      .filter($"gradelevel".isNotNull)
      .filter($"income".isNotNull)
      .filter($"special_education".isNotNull)
      .filter($"free_reduced_lunch_group".isNotNull)
      .filter($"min_math_unit_score".isNotNull)
      .filter($"avg_math_unit_score".isNotNull)
      .filter($"max_math_unit_score".isNotNull)
      .filter($"std_math_unit_score".isNotNull)
      .filter($"math_unit_n_1_ace".isNotNull)
      .filter($"math_unit_n_1_mastery".isNotNull)
      .filter($"math_unit_n_1_nomastery".isNotNull)
      .filter($"n_software_call".isNotNull)
      .filter($"n_hardware_call".isNotNull)
      .filter($"n_material_call".isNotNull)
      .filter($"avg_lesson_ontrack_status1".isNotNull)
      .filter($"std_lesson_ontrack_status1".isNotNull)
      .filter($"avg_lesson_ontrack_status2".isNotNull)
      .filter($"std_lesson_ontrack_status2".isNotNull)
      .filter($"num_active_con".isNotNull)
      .filter($"days_reg_recvd_school_start".isNotNull)
      .filter($"days_enroll_apv_school_start".isNotNull)
      .filter($"Flag".isNotNull)
      .toDF()

    //Model_Schema.describe().filter($"summary" === "count").show

    val Array(trainData, testData) = Model_Flag.randomSplit(Array(0.7, 0.3), seed = 1L)
    trainData.cache()
    testData.cache()

    val GradeOneHotEncoder = new OneHotEncoder()
      .setInputCol("gradelevel")
      .setOutputCol("gradelevelVector")

    val FamilyOneHotEncoder = new OneHotEncoder()
      .setInputCol("family_size")
      .setOutputCol("family_sizeVector")

    val FLunchOneHotEncoder = new OneHotEncoder()
      .setInputCol("free_reduced_lunch_group")
      .setOutputCol("free_reduced_lunch_groupVector")

    val sexIndexer = new StringIndexer()
      .setInputCol("special_education")
      .setOutputCol("special_educationIndex")

    val inputCols = Array("gradelevelVector",
      "special_educationIndex",
      "free_reduced_lunch_groupVector",
      "family_sizeVector",
      "income",
      "min_math_unit_score",
      "avg_math_unit_score",
      "max_math_unit_score",
      "std_math_unit_score",
      "math_unit_n_1_ace",
      "math_unit_n_1_mastery",
      "math_unit_n_1_nomastery",
      "n_software_call",
      "n_hardware_call",
      "n_material_call",
      "avg_lesson_ontrack_status1",
      "std_lesson_ontrack_status1",
      "avg_lesson_ontrack_status2",
      "std_lesson_ontrack_status2",
      "num_active_con",
      "days_reg_recvd_school_start",
      "days_enroll_apv_school_start")

    val assembler = new VectorAssembler()
      .setInputCols(inputCols)
      .setOutputCol("features")

    val classifier = new LogisticRegression()
      .setLabelCol("Flag")
      .setFeaturesCol("features")
      .setPredictionCol("Prediction")
      .setMaxIter(100)
      //.setRegParam(0.3)
      //.setElasticNetParam(0.8)

    val evaluator = new MulticlassClassificationEvaluator().
      setLabelCol("Flag")
      .setPredictionCol("Prediction")
      .setMetricName("accuracy")

    val pipeline = new Pipeline()
      .setStages(Array(GradeOneHotEncoder,
        FamilyOneHotEncoder,
        FLunchOneHotEncoder,
        sexIndexer,
        assembler,
        classifier))

    /*
    # Choose the best hyper-parameters using validation set
    */

    val regP = 0.01 to 0.1 by 0.01
    val ElaP = 0.1 to 0.9 by 0.1
    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.regParam, regP)
      .addGrid(classifier.elasticNetParam, ElaP)
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setTrainRatio(0.7)

    val validatorModel = trainValidationSplit.fit(trainData)
    val bestModel = validatorModel.bestModel
    println(bestModel.asInstanceOf[PipelineModel].stages.last.extractParamMap)

    val bestLRModel = bestModel.asInstanceOf[PipelineModel].stages.last.asInstanceOf[LogisticRegressionModel]
    println(s"Coefficients: ${bestLRModel.coefficients} Intercept: ${bestLRModel.intercept}")

    val trainPredictions =  bestModel.transform(trainData)
    val testPredictions = bestModel.transform(testData)

    val trainAccuracy = evaluator.evaluate(trainPredictions)
    println("Train Data Accuracy = " + trainAccuracy)

    val testAccuracy = evaluator.evaluate(testPredictions)
    println("Test Data Accuracy = " + testAccuracy)
  }
}
