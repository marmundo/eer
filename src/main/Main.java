package main;


import java.util.Random;
import weka.core.Instances;
import weka.core.WekaPackageManager;
import weka.classifiers.evaluation.EER;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import com.marcelodamasceno.util.ArffConector;

public class Main {

	/**
	 * @param args
	 */
	private String path="/home/marcelo/Área de Trabalho/Documentos-Windows/Google Drive/doutorado/projeto/dataset/Base de Toque/InterSession/";
	private String dataset="InterSession-User_1_Day_1_Horizontal.arff";

	public static void main(String[] args) {
		Main main=new Main();
		main.execute();
	}

	private void execute(){
		ArffConector conector=new ArffConector();		
		try {
			Instances data=conector.openDataSet(path+dataset);
			WekaPackageManager.loadPackages(false, false);
			Evaluation eval = new Evaluation(data);
			IBk ibk=new IBk(5);
			eval.crossValidateModel(ibk, data, 10, new Random(1));
			//System.out.println(Evaluation.getAllEvaluationMetricNames());
			//PackageManager.create().setPackageHome(new File("/home/marcelo/wekafiles/"));
			//PackageManager pack=PackageManager.create();
			//pack.setPackageHome(new File("/home/marcelo/wekafiles/"));
			//System.out.println(pack.getInstalledPackages());
			EER eer = (EER) eval.getPluginMetric("EER");
			//System.out.println(eer);
			//System.out.println(eer.getStatistic(""));
			//Evaluation evaluation = new Evaluation(trainInstances);
			//evaluation.evaluateModel(scheme, testInstances);
			
			//Using a modified way
			/*EER err=new EER();
			err.setBaseEvaluation(eval);*/
			System.out.println(eer.getStatistic(""));
			
			
			/*System.out.println(eval.toSummaryString());			
			ThresholdCurve curve=new ThresholdCurve();
			Instances rocPoints= curve.getCurve(eval.predictions());		
			EER eer=new EER(rocPoints);
			System.out.println("EER= "+eer.calculateEER()*100+"%");
			*/
			
		} catch (Exception e) {
			e.printStackTrace();
		}		
	}

}
