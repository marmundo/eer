package main.java.weka.classifiers.evaluation;

import java.util.Enumeration;
import java.util.LinkedList;
import java.util.List;
import weka.core.Instance;
import weka.core.Instances;

/**
 * @author Marcelo
 *Class to calculate the EER using points in the Threshold Curve (ROC)
 *
 *Example of usage:
 *IBk ibk=new IBk(5);
 *eval.crossValidateModel(ibk, data, 10, new Random(1));
 *ThresholdCurve curve=new ThresholdCurve();
 *Instances rocPoints= curve.getCurve(eval.predictions());		
 *EER eer=new EER(rocPoints);
 *System.out.println("EER= "+eer.calculateEER()*100+"%");
 */

public class EER {

	private Instances rocPoints;
	/**
	 * True Positive Points
	 */
	private List<Double> TPR = new LinkedList<Double>();
	
	/**
	 * False Positive Points
	 */
	private List<Double> FPR = new LinkedList<Double>();;

	public EER(Instances curvePoints){
		rocPoints=curvePoints;
	}
	/**
	 * Leaves only the True and False Positives Rates from the Thereshold Curve Instances
	 * @param rocPoints Instances generated from the getCurve method from the TheresholdCurve Class
	 * @return Instances with only True and False Positives Rates
	 */
	private Instances removeOptionalAttributes(Instances rocPoints){
		for (int i = 0; i < 4; i++) {
			rocPoints.deleteAttributeAt(0);
		}
		for (int i = 0; i < 7; i++) {
			rocPoints.deleteAttributeAt(2);
		}
		return rocPoints;
	}

	public double calculateEER(){
		rocPoints=removeOptionalAttributes(rocPoints);
		getFPRTPR(rocPoints);	
		LinkedList<Double> xy=getXequalsY();
		System.out.println("ERROR from ROC points (FPR,TPR) ("+xy.get(0)+","+xy.get(1)+")");
		LinkedList<Double> points=(getPointsXequalsY());
		/*# Extract the two points as (x) and (y), and find the point on the
		# line between x and y where the first and second elements of the
		# vector are equal.  Specifically, the line through x and y is:
		#   x + a*(y-x) for all a, and we want a such that
		#   x[1] + a*(y[1]-x[1]) = x[2] + a*(y[2]-x[2]) so
		#   a = (x[1] - x[2]) / (y[2]-x[2]-y[1]+x[1])*/
		double x1=points.get(0);
		double y1=points.get(1);
		double x2=points.get(2);
		double y2=points.get(3);
		double a=( x1 - x2 ) / ( y2 - x2 - y1 + x1 );
		System.out.println("x1,y1:"+x1+","+y1);
		System.out.println("x1,y1:"+x2+","+y2);
		System.out.println("a:"+a);
		double eer=x1 + a * ( y1 - x1 );
		return eer;
	}

	/**
	 * Method to feed the TruePositive and FalsePositive Lists
	 * @param rocPoints
	 */
	@SuppressWarnings("unchecked")
	private void getFPRTPR(Instances rocPoints){
		Enumeration<Instance> enu=rocPoints.enumerateInstances();
		while (enu.hasMoreElements()) {
			Instance instance = (Instance) enu.nextElement();
			FPR.add(instance.value(0));
			TPR.add(1-instance.value(1));
		}
	}

	/**
	 * Method to get the 2 points in the ROC Curve where X is aproximately equals Y, i.e., FPR=(1-TPR).
	 * @return 4 points: x1,y1,x2,y2. Where x1<y1 and x2>y2 and the difference between xi,yi is the minimum
	 */
	private LinkedList<Double> getPointsXequalsY(){
		double min=1;
		int index=1;
		for (int i = 0; i < FPR.size(); i++) {
			double diff=FPR.get(i)-TPR.get(i);
			if(Math.abs(diff)<min){
				min=diff;
				index=i;
			}
		}		
		LinkedList<Double> points = new LinkedList<Double>();
		points.add(FPR.get(index));
		points.add(TPR.get(index));

		double diff1=Math.abs(FPR.get(index-1)-TPR.get(index-1));
		double diff2=Math.abs(FPR.get(index+1)-TPR.get(index+1));
		if(diff1<diff2){
			points.add(FPR.get(index-1));
			points.add(TPR.get(index-1));
		}else{
			points.add(FPR.get(index+1));
			points.add(TPR.get(index+1));
		}
		return points;

	}

	private LinkedList<Double> getXequalsY(){
		double min=1;
		int index=1;
		for (int i = 0; i < FPR.size(); i++) {
			double diff=FPR.get(i)-TPR.get(i);
			if(Math.abs(diff)<min){
				min=diff;
				index=i;
			}
		}		
		LinkedList<Double> points = new LinkedList<Double>();
		points.add(FPR.get(index));
		points.add(TPR.get(index));		
		return points;		
	}
}
