import org.apache.spark.sql
import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.L1Updater

object POC {
  def main(Args: Array[String]): Unit = {
    val isLocal = true

    //// Starting Spark Session
    val sparkSession = if (isLocal) {
      SparkSession.builder
        .master("local[*]")
        .appName("K12-SparkPOC")
        .config("spark.sql.parquet.compression.codec", "gzip")
        .getOrCreate()
    } else {
      SparkSession.builder
        .appName("my-spark-app")
        .config("spark.some.config.option", "config-value")
        .getOrCreate()
    }
    import sparkSession.implicits._

    //// Loading data from CSV file
    val Data = sparkSession.read.option("header", "true")
      .option("charset", "UTF8")
      .option("delimiter", ",")
      .option("inferSchema", "true")
      .csv("./data/Data.csv")
    //Data.printSchema()

    //// Subseting data
    val Data_F = Data.select($"studentid",
      $"schoolid",
      $"retained_completed_sy",
      $"gradelevel",
      $"family_size",
      $"income",
      $"income_frequency",
      $"special_education",
      $"free_reduced_lunch_group",
      $"min_math_unit_score",
      $"avg_math_unit_score",
      $"max_math_unit_score",
      $"std_math_unit_score",
      $"math_unit_n_1_ace",
      $"math_unit_n_1_mastery",
      $"math_unit_n_1_nomastery",
      $"n_software_call"
    )
    Data_F.describe().show()

    //// Creating Endogenous variable
    val To_Model = Data_F.filter($"retained_completed_sy" === "RETAINED" ||
      $"retained_completed_sy" === "IN YEAR WITHDRAW" ||
      $"retained_completed_sy" === "BETWEEN SY WITHDRAW")
    val Model_Flag = To_Model.withColumn("Flag", when($"retained_completed_sy" === "RETAINED", 1.0).otherwise(0.0))
    Model_Flag.groupBy($"Flag").count().show()

    //// Creating Imputation values and Imputed Variables
    val Mean_Family = Model_Flag.select(mean($"family_size")).first()(0).asInstanceOf[Double]
    val Mean_Income = Model_Flag.select(mean($"income")).first()(0).asInstanceOf[Double]
    val Mean_Free = Model_Flag.select(mean($"free_reduced_lunch_group")).first()(0).asInstanceOf[Double]
    val Mean_MinMath = Model_Flag.select(mean($"min_math_unit_score")).first()(0).asInstanceOf[Double]
    val Mean_AvgMath = Model_Flag.select(mean($"avg_math_unit_score")).first()(0).asInstanceOf[Double]
    val Mean_MaxMath = Model_Flag.select(mean($"max_math_unit_score")).first()(0).asInstanceOf[Double]
    val Mean_StdMath = Model_Flag.select(mean($"std_math_unit_score")).first()(0).asInstanceOf[Double]
    val Mean_MathAce = Model_Flag.select(mean($"math_unit_n_1_ace")).first()(0).asInstanceOf[Double]
    val Mean_MathMastery = Model_Flag.select(mean($"math_unit_n_1_mastery")).first()(0).asInstanceOf[Double]
    val Mean_MathNoMastery = Model_Flag.select(mean($"math_unit_n_1_nomastery")).first()(0).asInstanceOf[Double]
    val Mean_Software = Model_Flag.select(mean($"n_software_call")).first()(0).asInstanceOf[Double]

    val Family_Imputed = Model_Flag.select("studentid", "family_size").na.fill(Mean_Family, Seq("family_size"))
    val Income_Imputed = Model_Flag.select("studentid", "income").na.fill(Mean_Income, Seq("income"))
    val Free_Imputed = Model_Flag.select("studentid", "free_reduced_lunch_group").na.fill(Mean_Free, Seq("free_reduced_lunch_group"))
    val MinMath_Imputed = Model_Flag.select("studentid", "min_math_unit_score").na.fill(Mean_MinMath, Seq("min_math_unit_score"))
    val AvgMath_Imputed = Model_Flag.select("studentid", "avg_math_unit_score").na.fill(Mean_AvgMath, Seq("avg_math_unit_score"))
    val MaxMath_Imputed = Model_Flag.select("studentid", "max_math_unit_score").na.fill(Mean_MaxMath, Seq("max_math_unit_score"))
    val StdMath_Imputed = Model_Flag.select("studentid", "std_math_unit_score").na.fill(Mean_StdMath, Seq("std_math_unit_score"))
    val MathAce_Imputed = Model_Flag.select("studentid", "math_unit_n_1_ace").na.fill(Mean_MathAce, Seq("math_unit_n_1_ace"))
    val MathMastery_Imputed = Model_Flag.select("studentid", "math_unit_n_1_mastery").na.fill(Mean_MathMastery, Seq("math_unit_n_1_mastery"))
    val MathNoMastery_Imputed = Model_Flag.select("studentid", "math_unit_n_1_nomastery").na.fill(Mean_MathNoMastery, Seq("math_unit_n_1_nomastery"))
    val Software_Imputed = Model_Flag.select("studentid", "n_software_call").na.fill(Mean_Software, Seq("n_software_call"))
    val Flag_Imputed = Model_Flag.select("studentid", "Flag").na.fill(0, Seq("Flag"))
    val To_Fit = Family_Imputed.join(Income_Imputed, "studentid")
      .join(Free_Imputed, "studentid")
      .join(MinMath_Imputed, "studentid")
      .join(AvgMath_Imputed, "studentid")
      .join(MaxMath_Imputed, "studentid")
      .join(StdMath_Imputed, "studentid")
      .join(MathAce_Imputed, "studentid")
      .join(MathMastery_Imputed, "studentid")
      .join(MathNoMastery_Imputed, "studentid")
      .join(Software_Imputed, "studentid")
      .join(Flag_Imputed, "studentid")
      .select(Income_Imputed("income"),
        MinMath_Imputed("min_math_unit_score"),
        AvgMath_Imputed("avg_math_unit_score"),
        MaxMath_Imputed("max_math_unit_score"),
        StdMath_Imputed("std_math_unit_score"),
        Flag_Imputed("Flag"))

    //// Creating Labeled Point to train the models
    val To_FitRDD = To_Fit.rdd
    val header = To_FitRDD.first
    val To_FitRDD_ = To_FitRDD.filter(_ (0) != header(0))
    val Labeled = To_FitRDD_.map { x =>
      LabeledPoint(x(5).asInstanceOf[Double], Vectors.dense(Array(x(0).asInstanceOf[Double],
        x(1).asInstanceOf[Double],
        x(2).asInstanceOf[Double],
        x(3).asInstanceOf[Double],
        x(4).asInstanceOf[Double])))
    }

    //// Splitting data into Training and Testing
    val splits = Labeled.randomSplit(Array(0.7, 0.3), seed = 11L)
    val training = splits(0)
    val test = splits(1)

    //// Training and Evaluating a Logistic Regression using LBFGS
    val lr_LBFGS = new LogisticRegressionWithLBFGS().setNumClasses(2).setIntercept(true)
    val model_LBFGS = lr_LBFGS.run(training)
    model_LBFGS.setThreshold(0.5)
    val predictionAndLabel_LBFGS = test.map(p => (model_LBFGS.predict(p.features), p.label))
    val accuracy_LBFGS = 1.0 * predictionAndLabel_LBFGS.filter(x => x._1 == x._2).count() / test.count()
    val metrics_LBFGS = new BinaryClassificationMetrics(predictionAndLabel_LBFGS)
    val AUC_LBFGS = metrics_LBFGS.areaUnderROC()
    println("Info for LR_LBFGS", model_LBFGS)
    println("Acurracy for LR_LBFGS", accuracy_LBFGS)
    println("AUC for LR_LBFGS", AUC_LBFGS)

    //// Training and Evaluating a Support Vector Machine using SGD
    val numIterations = 100
    val model_SVM = SVMWithSGD.train(training, numIterations)
    model_SVM.setThreshold(0.5)
    val predictionAndLabel_SVM = test.map(p => (model_SVM.predict(p.features), p.label))
    val accuracy_SVM = 1.0 * predictionAndLabel_SVM.filter(x => x._1 == x._2).count() / test.count()
    val metrics_SVM = new BinaryClassificationMetrics(predictionAndLabel_SVM)
    val AUC_SVM = metrics_SVM.areaUnderROC()
    println("Info for SVM_SGD", model_SVM)
    println("Accuracy for SVM_SGD", accuracy_SVM)
    println("AUC for SVM_SGD", AUC_SVM)

    //// Training and Evaluating a Support Vector Machine using SGD and Optimizing Features Weights using L1 Regularization
    val svmAlg = new SVMWithSGD()
    svmAlg.optimizer.
      setNumIterations(100).
      setRegParam(0.05).
      setUpdater(new L1Updater)
    val SVMSGD_L1 = svmAlg.run(training)
    model_SVM.setThreshold(0.5)
    val predictionAndLabel_SVML1 = test.map(p => (SVMSGD_L1.predict(p.features), p.label))
    val accuracy_SVML1 = 1.0 * predictionAndLabel_SVML1.filter(x => x._1 == x._2).count() / test.count()
    val metrics_SVML1 = new BinaryClassificationMetrics(predictionAndLabel_SVML1)
    val AUC_SVML1 = metrics_SVML1.areaUnderROC()
    println("Info for L1_SVMSGD", SVMSGD_L1)
    println("Accuracy for L1_SVMSGD", accuracy_SVML1)
    println("AUC for L1_SVMSGD", AUC_SVML1)
  }
}