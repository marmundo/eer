package main;

import java.util.Random;


import weka.core.Instances;
import weka.classifiers.evaluation.EER;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.classifiers.evaluation.ThresholdCurve;

import com.marcelodamasceno.util.ArffConector;

public class Main {

	/**
	 * @param args
	 */
	private String path="C:/Users/Marcelo/workspace/Arff Editor/";
	private String dataset="binary2.arff";

	public static void main(String[] args) {
		Main main=new Main();
		main.execute();
	}

	private void execute(){
		ArffConector conector=new ArffConector();
		Instances data=conector.openDataSet(path+dataset);
		IBk ibk=new IBk(5);
		//ibk.buildClassifier(data);
		Evaluation eval;
		try {
			eval = new Evaluation(data);
			eval.crossValidateModel(ibk, data, 10, new Random(1));
			//Evaluation evaluation = new Evaluation(trainInstances);
			//evaluation.evaluateModel(scheme, testInstances);
			System.out.println(eval.toSummaryString());			
			ThresholdCurve curve=new ThresholdCurve();
			Instances rocPoints= curve.getCurve(eval.predictions());		
			EER eer=new EER(rocPoints);
			System.out.println("EER= "+eer.calculateEER()*100+"%");
			
			
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}

}
