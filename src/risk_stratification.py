"""
Risk Stratification Module
Computes composite risk scores and stratifies patients into risk categories
based on multiple cardiac parameters and clinical factors.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskScore:
    """Data class for risk score results."""
    overall_score: float
    category: str
    contributing_factors: Dict[str, float]
    percentile: float
    recommendations: List[str]


class ClinicalRiskStratifier:
    """
    Comprehensive risk stratification system for cardiac patients.
    Computes multiple risk scores and provides integrated risk assessment.
    """
    
    def __init__(self):
        """Initialize risk stratifier with scoring criteria."""
        self._define_risk_criteria()
    
    def _define_risk_criteria(self):
        """Define risk scoring criteria and thresholds."""
        
        # Risk factor weights (normalized to sum to 1.0)
        self.risk_weights = {
            'age': 0.10,
            'ef_dysfunction': 0.25,
            'diastolic_dysfunction': 0.20,
            'lvh': 0.15,
            'lv_dilation': 0.10,
            'la_enlargement': 0.10,
            'valvular_disease': 0.10
        }
        
        # Age-based risk scoring
        self.age_risk = {
            'low': (0, 50),
            'moderate': (50, 65),
            'high': (65, 75),
            'very_high': (75, 120)
        }
        
        # Risk thresholds
        self.risk_categories = {
            'Low Risk': (0, 25),
            'Moderate Risk': (25, 50),
            'High Risk': (50, 75),
            'Very High Risk': (75, 100)
        }
        
        # Framingham-style risk factors
        self.framingham_factors = {
            'age_male': {50: 0, 55: 2, 60: 3, 65: 4, 70: 5, 75: 6, 80: 7},
            'age_female': {50: 0, 55: 2, 60: 4, 65: 5, 70: 6, 75: 7, 80: 8},
            'diabetes': 2,
            'smoking': 2,
            'hypertension': 1,
            'family_history': 1
        }
    
    def compute_cardiovascular_risk_score(self, 
                                         measurements: Dict[str, float],
                                         patient_info: Dict[str, Any],
                                         clinical_factors: Optional[Dict[str, Any]] = None) -> RiskScore:
        """
        Compute comprehensive cardiovascular risk score.
        
        Args:
            measurements: Cardiac measurements
            patient_info: Patient demographics (age, sex)
            clinical_factors: Additional clinical factors (diabetes, smoking, etc.)
            
        Returns:
            RiskScore object with detailed scoring
        """
        if clinical_factors is None:
            clinical_factors = {}
        
        risk_components = {}
        
        # 1. Age-based risk
        age = patient_info.get('age', 50)
        age_score = self._score_age_risk(age)
        risk_components['age'] = age_score * self.risk_weights['age']
        
        # 2. Systolic function risk (EF)
        ef = measurements.get('EF', 65)
        ef_score = self._score_ef_risk(ef)
        risk_components['ef_dysfunction'] = ef_score * self.risk_weights['ef_dysfunction']
        
        # 3. Diastolic dysfunction risk
        e_a_ratio = measurements.get('MV_E_A', measurements.get('E_A_ratio', None))
        e_e_prime = measurements.get('E_E_prime', None)
        dd_score = self._score_diastolic_risk(e_a_ratio, e_e_prime)
        risk_components['diastolic_dysfunction'] = dd_score * self.risk_weights['diastolic_dysfunction']
        
        # 4. LVH risk
        sex = patient_info.get('sex', 'M')
        lv_mass = measurements.get('LV_MASS', measurements.get('LV_mass', None))
        ivs_d = measurements.get('IVS_D', measurements.get('IVS_d', None))
        lvh_score = self._score_lvh_risk(lv_mass, ivs_d, sex)
        risk_components['lvh'] = lvh_score * self.risk_weights['lvh']
        
        # 5. LV dilation risk
        lvid_d = measurements.get('LVID_D', measurements.get('LVIDd', None))
        lv_dilation_score = self._score_lv_dilation_risk(lvid_d, sex)
        risk_components['lv_dilation'] = lv_dilation_score * self.risk_weights['lv_dilation']
        
        # 6. LA enlargement risk
        la_dimension = measurements.get('LA_DIMENSION', measurements.get('LA_dimension', None))
        la_score = self._score_la_risk(la_dimension, sex)
        risk_components['la_enlargement'] = la_score * self.risk_weights['la_enlargement']
        
        # 7. Valvular disease risk
        mr_grade = measurements.get('MR_grade', 0)
        ar_grade = measurements.get('AR_grade', 0)
        valvular_score = self._score_valvular_risk(mr_grade, ar_grade)
        risk_components['valvular_disease'] = valvular_score * self.risk_weights['valvular_disease']
        
        # Compute overall risk score (0-100)
        overall_score = sum(risk_components.values())
        
        # Determine risk category
        category = self._categorize_risk(overall_score)
        
        # Compute percentile (mock - would need population data)
        percentile = self._estimate_risk_percentile(overall_score, age, sex)
        
        # Generate recommendations
        recommendations = self._generate_risk_recommendations(
            overall_score, 
            risk_components, 
            clinical_factors
        )
        
        return RiskScore(
            overall_score=overall_score,
            category=category,
            contributing_factors=risk_components,
            percentile=percentile,
            recommendations=recommendations
        )
    
    def compute_heart_failure_risk(self,
                                   measurements: Dict[str, float],
                                   patient_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute heart failure risk score.
        Based on parameters like EF, BNP, symptoms.
        
        Args:
            measurements: Cardiac measurements
            patient_info: Patient demographics
            
        Returns:
            Heart failure risk assessment
        """
        ef = measurements.get('EF', 65)
        lvid_d = measurements.get('LVID_D', 4.5)
        la_dimension = measurements.get('LA_DIMENSION', 3.5)
        e_e_prime = measurements.get('E_E_prime', 8)
        age = patient_info.get('age', 50)
        
        # Score components (0-100)
        ef_component = max(0, (65 - ef) / 65 * 100) if ef < 65 else 0
        lv_size_component = max(0, (lvid_d - 5.5) / 5.5 * 100) if lvid_d > 5.5 else 0
        la_component = max(0, (la_dimension - 4.0) / 4.0 * 100) if la_dimension > 4.0 else 0
        filling_component = max(0, (e_e_prime - 13) / 13 * 100) if e_e_prime > 13 else 0
        age_component = max(0, (age - 65) / 65 * 100) if age > 65 else 0
        
        # Weighted score
        hf_score = (
            ef_component * 0.35 +
            lv_size_component * 0.20 +
            la_component * 0.15 +
            filling_component * 0.20 +
            age_component * 0.10
        )
        
        # Classify risk
        if hf_score < 20:
            hf_risk = "Low"
            hf_description = "Low risk of developing heart failure"
        elif hf_score < 40:
            hf_risk = "Moderate"
            hf_description = "Moderate risk - monitor closely"
        elif hf_score < 60:
            hf_risk = "High"
            hf_description = "High risk - consider early intervention"
        else:
            hf_risk = "Very High"
            hf_description = "Very high risk - aggressive management needed"
        
        return {
            'risk_score': hf_score,
            'risk_category': hf_risk,
            'description': hf_description,
            'components': {
                'ef': ef_component,
                'lv_size': lv_size_component,
                'la_size': la_component,
                'filling_pressure': filling_component,
                'age': age_component
            },
            'one_year_risk_percent': self._estimate_one_year_hf_risk(hf_score),
            'five_year_risk_percent': self._estimate_five_year_hf_risk(hf_score)
        }
    
    def compute_mortality_risk(self,
                              measurements: Dict[str, float],
                              patient_info: Dict[str, Any],
                              clinical_factors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute all-cause mortality risk.
        
        Args:
            measurements: Cardiac measurements
            patient_info: Patient demographics
            clinical_factors: Additional clinical factors
            
        Returns:
            Mortality risk assessment
        """
        if clinical_factors is None:
            clinical_factors = {}
        
        age = patient_info.get('age', 50)
        sex = patient_info.get('sex', 'M')
        ef = measurements.get('EF', 65)
        
        # Base mortality risk from age
        if age < 50:
            base_risk = 1.0
        elif age < 60:
            base_risk = 2.0
        elif age < 70:
            base_risk = 5.0
        elif age < 80:
            base_risk = 10.0
        else:
            base_risk = 20.0
        
        # Risk multipliers
        risk_multiplier = 1.0
        
        # EF impact
        if ef < 30:
            risk_multiplier *= 4.0
        elif ef < 40:
            risk_multiplier *= 2.5
        elif ef < 50:
            risk_multiplier *= 1.5
        
        # Clinical factors
        if clinical_factors.get('diabetes', False):
            risk_multiplier *= 1.5
        
        if clinical_factors.get('smoking', False):
            risk_multiplier *= 1.8
        
        if clinical_factors.get('ckd', False):  # Chronic kidney disease
            risk_multiplier *= 2.0
        
        # Compute risk
        one_year_risk = min(base_risk * risk_multiplier, 100)
        five_year_risk = min(one_year_risk * 3.5, 100)
        ten_year_risk = min(five_year_risk * 1.8, 100)
        
        return {
            'one_year_mortality': one_year_risk,
            'five_year_mortality': five_year_risk,
            'ten_year_mortality': ten_year_risk,
            'base_risk': base_risk,
            'risk_multiplier': risk_multiplier,
            'risk_category': self._categorize_mortality_risk(five_year_risk)
        }
    
    def compute_composite_risk_index(self,
                                    measurements: Dict[str, float],
                                    patient_info: Dict[str, Any],
                                    clinical_factors: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Compute comprehensive composite risk index combining multiple scores.
        
        Args:
            measurements: Cardiac measurements
            patient_info: Patient demographics
            clinical_factors: Additional clinical factors
            
        Returns:
            Composite risk assessment
        """
        # Compute individual risk scores
        cv_risk = self.compute_cardiovascular_risk_score(
            measurements, patient_info, clinical_factors
        )
        
        hf_risk = self.compute_heart_failure_risk(measurements, patient_info)
        
        mortality_risk = self.compute_mortality_risk(
            measurements, patient_info, clinical_factors
        )
        
        # Compute composite index (weighted average)
        composite_score = (
            cv_risk.overall_score * 0.4 +
            hf_risk['risk_score'] * 0.35 +
            mortality_risk['five_year_mortality'] * 0.25
        )
        
        # Determine overall risk tier
        risk_tier = self._categorize_risk(composite_score)
        
        # Priority recommendations
        all_recommendations = cv_risk.recommendations.copy()
        
        if hf_risk['risk_category'] in ['High', 'Very High']:
            all_recommendations.append("Consider heart failure preventive therapy")
        
        if mortality_risk['five_year_mortality'] > 20:
            all_recommendations.append("Aggressive risk factor modification indicated")
        
        return {
            'composite_score': composite_score,
            'risk_tier': risk_tier,
            'cardiovascular_risk': {
                'score': cv_risk.overall_score,
                'category': cv_risk.category,
                'percentile': cv_risk.percentile
            },
            'heart_failure_risk': {
                'score': hf_risk['risk_score'],
                'category': hf_risk['risk_category'],
                'one_year': hf_risk['one_year_risk_percent'],
                'five_year': hf_risk['five_year_risk_percent']
            },
            'mortality_risk': {
                'one_year': mortality_risk['one_year_mortality'],
                'five_year': mortality_risk['five_year_mortality'],
                'ten_year': mortality_risk['ten_year_mortality'],
                'category': mortality_risk['risk_category']
            },
            'recommendations': all_recommendations,
            'follow_up_interval': self._recommend_follow_up_interval(composite_score)
        }
    
    # Helper scoring functions
    
    def _score_age_risk(self, age: float) -> float:
        """Score age-related risk (0-100)."""
        if age < 50:
            return 10
        elif age < 60:
            return 25
        elif age < 70:
            return 50
        elif age < 80:
            return 75
        else:
            return 90
    
    def _score_ef_risk(self, ef: float) -> float:
        """Score EF-related risk (0-100)."""
        if ef >= 55:
            return 0
        elif ef >= 45:
            return 25
        elif ef >= 35:
            return 50
        elif ef >= 25:
            return 75
        else:
            return 100
    
    def _score_diastolic_risk(self, e_a_ratio: Optional[float], 
                             e_e_prime: Optional[float]) -> float:
        """Score diastolic dysfunction risk (0-100)."""
        score = 0
        count = 0
        
        if e_a_ratio is not None:
            count += 1
            if e_a_ratio > 2.0:  # Restrictive
                score += 100
            elif e_a_ratio < 0.8:  # Impaired relaxation
                score += 40
            else:
                score += 0
        
        if e_e_prime is not None:
            count += 1
            if e_e_prime > 14:
                score += 70
            elif e_e_prime > 10:
                score += 40
            else:
                score += 0
        
        return score / count if count > 0 else 0
    
    def _score_lvh_risk(self, lv_mass: Optional[float], 
                       ivs_d: Optional[float], sex: str) -> float:
        """Score LVH-related risk (0-100)."""
        score = 0
        count = 0
        
        if lv_mass is not None:
            count += 1
            threshold = 224 if sex.upper() == 'M' else 162
            if lv_mass > threshold * 1.5:
                score += 80
            elif lv_mass > threshold * 1.3:
                score += 60
            elif lv_mass > threshold:
                score += 30
        
        if ivs_d is not None:
            count += 1
            if ivs_d > 1.5:
                score += 70
            elif ivs_d > 1.3:
                score += 50
            elif ivs_d > 1.1:
                score += 25
        
        return score / count if count > 0 else 0
    
    def _score_lv_dilation_risk(self, lvid_d: Optional[float], sex: str) -> float:
        """Score LV dilation risk (0-100)."""
        if lvid_d is None:
            return 0
        
        threshold = 5.9 if sex.upper() == 'M' else 5.3
        
        if lvid_d > threshold * 1.3:
            return 90
        elif lvid_d > threshold * 1.15:
            return 60
        elif lvid_d > threshold:
            return 30
        else:
            return 0
    
    def _score_la_risk(self, la_dimension: Optional[float], sex: str) -> float:
        """Score LA enlargement risk (0-100)."""
        if la_dimension is None:
            return 0
        
        threshold = 4.0 if sex.upper() == 'M' else 3.8
        
        if la_dimension > threshold * 1.4:
            return 85
        elif la_dimension > threshold * 1.2:
            return 60
        elif la_dimension > threshold:
            return 30
        else:
            return 0
    
    def _score_valvular_risk(self, mr_grade: int, ar_grade: int) -> float:
        """Score valvular disease risk (0-100)."""
        max_grade = max(mr_grade, ar_grade)
        
        if max_grade >= 3:
            return 80
        elif max_grade == 2:
            return 50
        elif max_grade == 1:
            return 20
        else:
            return 0
    
    def _categorize_risk(self, score: float) -> str:
        """Categorize risk based on score."""
        for category, (min_val, max_val) in self.risk_categories.items():
            if min_val <= score < max_val:
                return category
        return "Very High Risk"
    
    def _categorize_mortality_risk(self, five_year_risk: float) -> str:
        """Categorize mortality risk."""
        if five_year_risk < 5:
            return "Low"
        elif five_year_risk < 15:
            return "Moderate"
        elif five_year_risk < 30:
            return "High"
        else:
            return "Very High"
    
    def _estimate_risk_percentile(self, score: float, age: int, sex: str) -> float:
        """Estimate risk percentile compared to population."""
        # Mock percentile calculation
        # In practice, would use population reference data
        base_percentile = (score / 100) * 95
        
        # Adjust for age
        if age > 70:
            base_percentile = max(base_percentile - 10, 0)
        elif age < 50:
            base_percentile = min(base_percentile + 10, 100)
        
        return base_percentile
    
    def _estimate_one_year_hf_risk(self, hf_score: float) -> float:
        """Estimate 1-year heart failure risk percentage."""
        return min(hf_score * 0.15, 25)
    
    def _estimate_five_year_hf_risk(self, hf_score: float) -> float:
        """Estimate 5-year heart failure risk percentage."""
        return min(hf_score * 0.5, 75)
    
    def _generate_risk_recommendations(self,
                                      overall_score: float,
                                      risk_components: Dict[str, float],
                                      clinical_factors: Dict[str, Any]) -> List[str]:
        """Generate personalized recommendations based on risk profile."""
        recommendations = []
        
        # Top contributing factors
        sorted_components = sorted(risk_components.items(), 
                                  key=lambda x: x[1], 
                                  reverse=True)
        
        top_factor = sorted_components[0][0]
        
        # General recommendations
        if overall_score >= 75:
            recommendations.append("Immediate cardiology consultation recommended")
            recommendations.append("Aggressive risk factor modification essential")
        elif overall_score >= 50:
            recommendations.append("Cardiology follow-up within 1-3 months")
            recommendations.append("Intensive lifestyle modifications needed")
        elif overall_score >= 25:
            recommendations.append("Regular cardiac monitoring advised")
            recommendations.append("Optimize blood pressure and lipid control")
        else:
            recommendations.append("Continue routine preventive care")
            recommendations.append("Maintain healthy lifestyle")
        
        # Specific recommendations based on top risk factor
        if 'ef_dysfunction' in top_factor:
            recommendations.append("Consider ACE inhibitors or beta-blockers")
        elif 'lvh' in top_factor:
            recommendations.append("Aggressive blood pressure control indicated")
        elif 'diastolic' in top_factor:
            recommendations.append("Volume management and rate control important")
        
        # Clinical factor recommendations
        if clinical_factors.get('diabetes', False):
            recommendations.append("Optimize glycemic control (HbA1c < 7%)")
        
        if clinical_factors.get('smoking', False):
            recommendations.append("Smoking cessation is critical")
        
        return recommendations
    
    def _recommend_follow_up_interval(self, composite_score: float) -> str:
        """Recommend follow-up interval based on risk."""
        if composite_score >= 75:
            return "1-3 months"
        elif composite_score >= 50:
            return "3-6 months"
        elif composite_score >= 25:
            return "6-12 months"
        else:
            return "12-24 months"
    
    def plot_risk_dashboard(self,
                           risk_assessment: Dict[str, Any],
                           patient_info: Dict[str, Any],
                           save_path: Optional[str] = None):
        """
        Create comprehensive risk stratification dashboard.
        
        Args:
            risk_assessment: Output from compute_composite_risk_index
            patient_info: Patient information
            save_path: Path to save plot
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
        
        # Title with patient info
        patient_text = f"Age: {patient_info.get('age', 'N/A')} | Sex: {patient_info.get('sex', 'N/A')}"
        fig.suptitle(f'Comprehensive Risk Stratification Dashboard\n{patient_text}', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Composite risk gauge
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_composite_gauge(ax1, risk_assessment['composite_score'])
        
        # 2. Individual risk scores
        ax2 = fig.add_subplot(gs[1, :2])
        self._plot_risk_scores(ax2, risk_assessment)
        
        # 3. Risk tier badge
        ax3 = fig.add_subplot(gs[1, 2])
        self._plot_risk_tier(ax3, risk_assessment['risk_tier'])
        
        # 4. Heart failure risk over time
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_hf_risk_timeline(ax4, risk_assessment['heart_failure_risk'])
        
        # 5. Mortality risk over time
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_mortality_timeline(ax5, risk_assessment['mortality_risk'])
        
        # 6. Risk components radar
        ax6 = fig.add_subplot(gs[2, 2], projection='polar')
        self._plot_risk_radar(ax6, risk_assessment)
        
        # 7. Recommendations
        ax7 = fig.add_subplot(gs[3, :])
        self._plot_risk_recommendations(ax7, risk_assessment['recommendations'],
                                       risk_assessment['follow_up_interval'])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved risk dashboard to {save_path}")
        
        plt.show()
    
    def _plot_composite_gauge(self, ax, score: float):
        """Plot composite risk gauge."""
        # Similar to severity gauge
        theta = np.linspace(0, np.pi, 100)
        colors = ['green', 'yellow', 'orange', 'red']
        boundaries = [0, 25, 50, 75, 100]
        
        for i in range(len(colors)):
            start_angle = boundaries[i] / 100 * np.pi
            end_angle = boundaries[i+1] / 100 * np.pi
            theta_segment = np.linspace(start_angle, end_angle, 25)
            ax.fill_between(theta_segment, 0, 1, color=colors[i], alpha=0.3)
        
        # Needle
        score_angle = score / 100 * np.pi
        ax.plot([score_angle, score_angle], [0, 0.9], 'k-', linewidth=3)
        ax.plot(score_angle, 0.9, 'ko', markersize=10)
        
        ax.set_ylim(0, 1)
        ax.set_xlim(0, np.pi)
        ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
        ax.set_xticklabels(['0\nLow', '25', '50\nHigh', '75', '100\nVery High'])
        ax.set_yticks([])
        ax.set_title(f'Composite Risk Score: {score:.1f}', fontsize=16, fontweight='bold')
    
    def _plot_risk_scores(self, ax, risk_assessment: Dict):
        """Plot individual risk component scores."""
        categories = ['Cardiovascular\nRisk', 'Heart Failure\nRisk', 'Mortality\nRisk\n(5-year)']
        scores = [
            risk_assessment['cardiovascular_risk']['score'],
            risk_assessment['heart_failure_risk']['score'],
            risk_assessment['mortality_risk']['five_year']
        ]
        colors = []
        
        for score in scores:
            if score < 25:
                colors.append('green')
            elif score < 50:
                colors.append('yellow')
            elif score < 75:
                colors.append('orange')
            else:
                colors.append('red')
        
        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel('Risk Score', fontsize=12)
        ax.set_ylim(0, 110)
        ax.set_title('Individual Risk Components', fontsize=14, fontweight='bold')
        ax.axhline(y=50, color='orange', linestyle='--', alpha=0.5, label='High Risk Threshold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{score:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    def _plot_risk_tier(self, ax, tier: str):
        """Plot risk tier badge."""
        ax.axis('off')
        colors = {'Low Risk': 'green', 'Moderate Risk': 'yellow', 
                 'High Risk': 'orange', 'Very High Risk': 'red'}
        color = colors.get(tier, 'gray')
        
        circle = plt.Circle((0.5, 0.5), 0.35, color=color, alpha=0.3)
        ax.add_patch(circle)
        ax.text(0.5, 0.5, tier, ha='center', va='center', fontsize=16, 
               fontweight='bold', wrap=True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Risk Tier', fontsize=14, fontweight='bold')
    
    def _plot_hf_risk_timeline(self, ax, hf_risk: Dict):
        """Plot heart failure risk timeline."""
        years = [1, 5]
        risks = [hf_risk['one_year'], hf_risk['five_year']]
        
        ax.plot(years, risks, marker='o', linewidth=3, markersize=10, color='steelblue')
        ax.fill_between(years, 0, risks, alpha=0.3, color='steelblue')
        ax.set_xlabel('Years', fontsize=11)
        ax.set_ylabel('Risk (%)', fontsize=11)
        ax.set_title('Heart Failure Risk Timeline', fontsize=12, fontweight='bold')
        ax.set_xticks(years)
        ax.set_ylim(0, max(risks) * 1.2 if risks else 10)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for year, risk in zip(years, risks):
            ax.text(year, risk + 1, f'{risk:.1f}%', ha='center', fontweight='bold')
    
    def _plot_mortality_timeline(self, ax, mortality_risk: Dict):
        """Plot mortality risk timeline."""
        years = [1, 5, 10]
        risks = [
            mortality_risk['one_year'],
            mortality_risk['five_year'],
            mortality_risk['ten_year']
        ]
        
        ax.plot(years, risks, marker='s', linewidth=3, markersize=10, color='crimson')
        ax.fill_between(years, 0, risks, alpha=0.3, color='crimson')
        ax.set_xlabel('Years', fontsize=11)
        ax.set_ylabel('Risk (%)', fontsize=11)
        ax.set_title('Mortality Risk Timeline', fontsize=12, fontweight='bold')
        ax.set_xticks(years)
        ax.set_ylim(0, max(risks) * 1.2 if risks else 10)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for year, risk in zip(years, risks):
            ax.text(year, risk + 1, f'{risk:.1f}%', ha='center', fontweight='bold')
    
    def _plot_risk_radar(self, ax, risk_assessment: Dict):
        """Plot risk components radar chart."""
        categories = ['CV Risk', 'HF Risk', 'Mortality']
        values = [
            risk_assessment['cardiovascular_risk']['score'],
            risk_assessment['heart_failure_risk']['score'],
            risk_assessment['mortality_risk']['five_year']
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # Complete the circle
        angles += angles[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, color='darkblue')
        ax.fill(angles, values, alpha=0.25, color='darkblue')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title('Risk Profile Radar', fontsize=12, fontweight='bold', pad=20)
        ax.grid(True)
    
    def _plot_risk_recommendations(self, ax, recommendations: List[str], 
                                  follow_up: str):
        """Plot recommendations."""
        ax.axis('off')
        
        text = "🔔 CLINICAL RECOMMENDATIONS\n\n"
        text += "\n".join([f"• {rec}" for rec in recommendations])
        text += f"\n\n📅 Recommended Follow-up: {follow_up}"
        
        ax.text(0.05, 0.95, text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
               family='monospace')


if __name__ == "__main__":
    print("Risk Stratification Module")
    print("=" * 50)
    print("Comprehensive risk assessment including:")
    print("- Cardiovascular risk scoring")
    print("- Heart failure risk prediction")
    print("- Mortality risk estimation")
    print("- Composite risk index")
    print("- Personalized recommendations")
