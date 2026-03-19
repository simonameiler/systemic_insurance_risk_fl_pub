"""
Comprehensive analysis of Emanuel TC Monte Carlo runs.
Evaluates loss composition, institutional stress, insured/uninsured ratios, and nonlinearities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

class EmanuelAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.scenarios = {}
        self.summary_stats = []
        
    def load_scenarios(self):
        """Load all Emanuel MC run scenarios."""
        print("Loading Emanuel scenarios...")
        
        # Get all Emanuel scenario directories
        emanuel_dirs = [d for d in self.results_dir.glob("emanuel_*") if d.is_dir()]
        
        for scenario_dir in sorted(emanuel_dirs):
            scenario_name = scenario_dir.name
            
            # Load iterations data
            iterations_file = scenario_dir / "iterations.csv"
            return_period_file = scenario_dir / "return_period_summary.csv"
            config_file = scenario_dir / "run_config.json"
            
            if not all([iterations_file.exists(), return_period_file.exists(), config_file.exists()]):
                print(f"  Skipping {scenario_name}: missing files")
                continue
            
            try:
                iterations_df = pd.read_csv(iterations_file)
                return_period_df = pd.read_csv(return_period_file)
                
                with open(config_file, 'r') as f:
                    config = json.load(f)
                
                self.scenarios[scenario_name] = {
                    'iterations': iterations_df,
                    'return_periods': return_period_df,
                    'config': config,
                    'dir': scenario_dir
                }
                
                print(f"  Loaded: {scenario_name}")
                
            except Exception as e:
                print(f"  Error loading {scenario_name}: {e}")
        
        print(f"\nTotal scenarios loaded: {len(self.scenarios)}")
        return self
    
    def parse_scenario_name(self, scenario_name):
        """Parse scenario name to extract components."""
        parts = scenario_name.replace("emanuel_", "").split("_")
        
        # Extract model and scenario
        model = parts[0] if parts else "unknown"
        
        # Detect climate scenario
        climate_scenario = "baseline"
        if "20thcal" in scenario_name:
            climate_scenario = "20th_century"
        elif "ssp245_2cal" in scenario_name:
            climate_scenario = "ssp245_2100"
        elif "ssp245cal" in scenario_name:
            climate_scenario = "ssp245_2050"
        elif "ssp585_2cal" in scenario_name:
            climate_scenario = "ssp585_2100"
        elif "ssp585cal" in scenario_name:
            climate_scenario = "ssp585_2050"
        
        # Detect policy
        policy_type = "baseline"
        policy_severity = "baseline"
        
        if "building_codes" in scenario_name:
            policy_type = "building_codes"
            if "major" in scenario_name:
                policy_severity = "major"
            elif "minor" in scenario_name:
                policy_severity = "minor"
            elif "moderate" in scenario_name:
                policy_severity = "moderate"
            elif "extreme" in scenario_name:
                policy_severity = "extreme"
        elif "market_exit" in scenario_name:
            policy_type = "market_exit"
            if "moderate" in scenario_name:
                policy_severity = "moderate"
            elif "major" in scenario_name:
                policy_severity = "major"
        elif "penetration" in scenario_name:
            policy_type = "penetration"
            if "major" in scenario_name:
                policy_severity = "major"
            elif "moderate" in scenario_name:
                policy_severity = "moderate"
        
        return {
            'model': model,
            'climate_scenario': climate_scenario,
            'policy_type': policy_type,
            'policy_severity': policy_severity
        }
    
    def compute_summary_statistics(self):
        """Compute comprehensive summary statistics for all scenarios."""
        print("\nComputing summary statistics...")
        
        for scenario_name, data in self.scenarios.items():
            df = data['iterations']
            parsed = self.parse_scenario_name(scenario_name)
            
            # Filter out zero-damage years for certain statistics
            damage_years = df[df['total_damage_usd'] > 0]
            
            # Basic damage statistics
            stats = {
                'scenario': scenario_name,
                'model': parsed['model'],
                'climate': parsed['climate_scenario'],
                'policy': parsed['policy_type'],
                'severity': parsed['policy_severity'],
                
                # Total damage
                'total_damage_mean': df['total_damage_usd'].mean(),
                'total_damage_median': df['total_damage_usd'].median(),
                'total_damage_std': df['total_damage_usd'].std(),
                'total_damage_99th': df['total_damage_usd'].quantile(0.99),
                
                # Wind vs flood damage
                'wind_damage_mean': df['wind_total_usd'].mean(),
                'water_damage_mean': df['water_total_usd'].mean(),
                'wind_share': df['wind_total_usd'].mean() / (df['total_damage_usd'].mean() + 1e-10),
                
                # Insurance coverage (wind)
                'wind_insured_private_mean': df['wind_insured_private_usd'].mean(),
                'wind_insured_citizens_mean': df['wind_insured_citizens_usd'].mean(),
                'wind_underinsured_mean': df['wind_underinsured_usd'].mean(),
                'wind_uninsured_mean': df['wind_uninsured_usd'].mean(),
                
                # Insurance penetration rates
                'wind_private_pct': df['wind_insured_private_pct'].mean(),
                'wind_citizens_pct': df['wind_insured_citizens_pct'].mean(),
                'wind_underinsured_pct': df['wind_underinsured_pct'].mean(),
                'wind_uninsured_pct': df['wind_uninsured_pct'].mean(),
                
                # Flood insurance
                'flood_insured_mean': df['flood_insured_capped_usd'].mean(),
                'flood_underinsured_mean': df['flood_un_derinsured_usd'].mean(),
                'flood_insured_pct': df['flood_insured_capped_pct'].mean(),
                'flood_underinsured_pct': df['flood_un_derinsured_pct'].mean(),
                
                # Institutional stress - defaults
                'defaults_mean': df['defaults_post'].mean(),
                'defaults_median': df['defaults_post'].median(),
                'defaults_max': df['defaults_post'].max(),
                'prob_any_default': (df['defaults_post'] > 0).mean(),
                'prob_5plus_defaults': (df['defaults_post'] >= 5).mean(),
                'prob_10plus_defaults': (df['defaults_post'] >= 10).mean(),
                
                # FIGA stress
                'figa_deficit_mean': df['figa_residual_deficit_usd'].mean(),
                'figa_deficit_median': df['figa_residual_deficit_usd'].median(),
                'figa_deficit_max': df['figa_residual_deficit_usd'].max(),
                'prob_figa_deficit': (df['figa_residual_deficit_usd'] > 0).mean(),
                
                # Citizens stress
                'citizens_deficit_mean': df['citizens_residual_deficit_usd'].mean(),
                'citizens_deficit_median': df['citizens_residual_deficit_usd'].median(),
                'citizens_deficit_max': df['citizens_residual_deficit_usd'].max(),
                'prob_citizens_deficit': (df['citizens_residual_deficit_usd'] > 0).mean(),
                'citizens_tier1_mean': df['citizens_tier1_usd'].mean(),
                'citizens_tier2_mean': df['citizens_tier2_usd'].mean(),
                
                # NFIP stress
                'nfip_borrowed_mean': df['nfip_borrowed_usd'].mean(),
                'nfip_borrowed_median': df['nfip_borrowed_usd'].median(),
                'nfip_borrowed_max': df['nfip_borrowed_usd'].max(),
                'prob_nfip_borrowing': (df['nfip_borrowed_usd'] > 0).mean(),
                'nfip_claims_paid_mean': df['nfip_claims_paid_usd'].mean(),
                
                # FHCF stress
                'fhcf_binding_prob': df['fhcf_cap_binding'].mean(),
                'fhcf_shortfall_mean': df['fhcf_shortfall_usd'].mean(),
                'fhcf_recovery_mean': df['fhcf_total_postcap_usd'].mean(),
                
                # Public burden (total government exposure)
                'public_burden_mean': (
                    df['citizens_residual_deficit_usd'] + 
                    df['figa_residual_deficit_usd'] + 
                    df['nfip_borrowed_usd']
                ).mean(),
                'public_burden_median': (
                    df['citizens_residual_deficit_usd'] + 
                    df['figa_residual_deficit_usd'] + 
                    df['nfip_borrowed_usd']
                ).median(),
                
                # Total uninsured/underinsured burden
                'total_uninsured_mean': (
                    df['wind_uninsured_usd'] + 
                    df['wind_underinsured_usd'] + 
                    df['flood_un_derinsured_usd']
                ).mean(),
                
                # Nonlinearity metrics
                'damage_cv': df['total_damage_usd'].std() / (df['total_damage_usd'].mean() + 1e-10),
                'defaults_damage_ratio': df['defaults_post'].mean() / (df['total_damage_usd'].mean() / 1e9 + 1e-10),
            }
            
            self.summary_stats.append(stats)
        
        self.summary_df = pd.DataFrame(self.summary_stats)
        print(f"Computed statistics for {len(self.summary_stats)} scenarios")
        return self
    
    def analyze_loss_composition(self):
        """Analyze how loss composition changes across scenarios."""
        print("\n" + "="*80)
        print("LOSS COMPOSITION ANALYSIS")
        print("="*80)
        
        # Group by climate scenario
        for climate in self.summary_df['climate'].unique():
            climate_data = self.summary_df[
                (self.summary_df['climate'] == climate) & 
                (self.summary_df['policy'] == 'baseline')
            ]
            
            if len(climate_data) == 0:
                continue
            
            print(f"\n{climate.upper()} - Baseline Policy:")
            print(f"  Number of models: {len(climate_data)}")
            print(f"  Total damage (mean): ${climate_data['total_damage_mean'].mean()/1e9:.2f}B ± ${climate_data['total_damage_mean'].std()/1e9:.2f}B")
            print(f"  Wind damage share: {climate_data['wind_share'].mean()*100:.1f}% ± {climate_data['wind_share'].std()*100:.1f}%")
            print(f"  Water damage share: {(1-climate_data['wind_share']).mean()*100:.1f}%")
            
            # Insurance breakdown
            total_wind = climate_data['wind_damage_mean'].mean()
            print(f"\n  Wind damage composition:")
            print(f"    Private insurance: {climate_data['wind_private_pct'].mean():.1f}%")
            print(f"    Citizens insurance: {climate_data['wind_citizens_pct'].mean():.1f}%")
            print(f"    Underinsured: {climate_data['wind_underinsured_pct'].mean():.1f}%")
            print(f"    Uninsured: {climate_data['wind_uninsured_pct'].mean():.1f}%")
            
            print(f"\n  Flood damage composition:")
            print(f"    NFIP insured: {climate_data['flood_insured_pct'].mean():.1f}%")
            print(f"    Underinsured: {climate_data['flood_underinsured_pct'].mean():.1f}%")
    
    def analyze_institutional_stress(self):
        """Analyze institutional stress indicators."""
        print("\n" + "="*80)
        print("INSTITUTIONAL STRESS ANALYSIS")
        print("="*80)
        
        baseline_data = self.summary_df[
            (self.summary_df['policy'] == 'baseline') & 
            (self.summary_df['climate'] == 'baseline')
        ]
        
        if len(baseline_data) > 0:
            print("\nBASELINE SCENARIOS (ERA5, no climate change):")
            print(f"  Mean defaults per simulation: {baseline_data['defaults_mean'].mean():.2f}")
            print(f"  Probability of any default: {baseline_data['prob_any_default'].mean()*100:.1f}%")
            print(f"  Probability of 10+ defaults: {baseline_data['prob_10plus_defaults'].mean()*100:.1f}%")
            print(f"  Mean FIGA deficit: ${baseline_data['figa_deficit_mean'].mean()/1e9:.2f}B")
            print(f"  Probability of FIGA deficit: {baseline_data['prob_figa_deficit'].mean()*100:.1f}%")
            print(f"  Mean Citizens deficit: ${baseline_data['citizens_deficit_mean'].mean()/1e9:.2f}B")
            print(f"  Probability of Citizens deficit: {baseline_data['prob_citizens_deficit'].mean()*100:.1f}%")
            print(f"  Mean NFIP borrowing: ${baseline_data['nfip_borrowed_mean'].mean()/1e9:.2f}B")
            print(f"  Probability of NFIP borrowing: {baseline_data['prob_nfip_borrowing'].mean()*100:.1f}%")
            print(f"  FHCF cap binding probability: {baseline_data['fhcf_binding_prob'].mean()*100:.1f}%")
        
        # Compare across climate scenarios
        print("\nCOMPARISON ACROSS CLIMATE SCENARIOS:")
        for climate in sorted(self.summary_df['climate'].unique()):
            climate_data = self.summary_df[
                (self.summary_df['climate'] == climate) & 
                (self.summary_df['policy'] == 'baseline')
            ]
            
            if len(climate_data) == 0:
                continue
            
            print(f"\n  {climate}:")
            print(f"    Defaults: {climate_data['defaults_mean'].mean():.2f}")
            print(f"    Public burden: ${climate_data['public_burden_mean'].mean()/1e9:.2f}B")
            print(f"    FIGA deficit prob: {climate_data['prob_figa_deficit'].mean()*100:.1f}%")
            print(f"    Citizens deficit prob: {climate_data['prob_citizens_deficit'].mean()*100:.1f}%")
    
    def analyze_policy_effects(self):
        """Analyze how different policies affect outcomes."""
        print("\n" + "="*80)
        print("POLICY INTERVENTION ANALYSIS")
        print("="*80)
        
        baseline_ref = self.summary_df[
            (self.summary_df['model'] == 'era5') & 
            (self.summary_df['climate'] == 'baseline') &
            (self.summary_df['policy'] == 'baseline')
        ]
        
        if len(baseline_ref) == 0:
            print("No ERA5 baseline reference found")
            return
        
        baseline_ref = baseline_ref.iloc[0]
        
        # Analyze each policy type
        for policy_type in ['building_codes', 'market_exit', 'penetration']:
            policy_data = self.summary_df[
                (self.summary_df['model'] == 'era5') & 
                (self.summary_df['policy'] == policy_type)
            ]
            
            if len(policy_data) == 0:
                continue
            
            print(f"\n{policy_type.upper().replace('_', ' ')}:")
            for _, row in policy_data.iterrows():
                severity = row['severity']
                damage_change = (row['total_damage_mean'] - baseline_ref['total_damage_mean']) / baseline_ref['total_damage_mean'] * 100
                defaults_change = row['defaults_mean'] - baseline_ref['defaults_mean']
                public_burden_change = (row['public_burden_mean'] - baseline_ref['public_burden_mean']) / (baseline_ref['public_burden_mean'] + 1e-10) * 100
                
                print(f"  {severity}:")
                print(f"    Damage change: {damage_change:+.1f}%")
                print(f"    Defaults change: {defaults_change:+.1f}")
                print(f"    Public burden change: {public_burden_change:+.1f}%")
                print(f"    Uninsured share: {row['wind_uninsured_pct']:.1f}%")
    
    def check_nonlinearities(self):
        """Check for nonlinear relationships in the data."""
        print("\n" + "="*80)
        print("NONLINEARITY ANALYSIS")
        print("="*80)
        
        # Focus on baseline policy scenarios
        baseline_data = self.summary_df[self.summary_df['policy'] == 'baseline'].copy()
        
        if len(baseline_data) < 5:
            print("Insufficient data for nonlinearity analysis")
            return
        
        # Calculate ratios
        baseline_data['public_burden_per_damage'] = (
            baseline_data['public_burden_mean'] / 
            (baseline_data['total_damage_mean'] + 1e-10) * 100
        )
        
        baseline_data['defaults_per_billion_damage'] = (
            baseline_data['defaults_mean'] / 
            (baseline_data['total_damage_mean'] / 1e9 + 1e-10)
        )
        
        print("\nPUBLIC BURDEN SENSITIVITY:")
        print(f"  Public burden as % of total damage:")
        print(f"    Min: {baseline_data['public_burden_per_damage'].min():.2f}%")
        print(f"    Mean: {baseline_data['public_burden_per_damage'].mean():.2f}%")
        print(f"    Max: {baseline_data['public_burden_per_damage'].max():.2f}%")
        print(f"    Std: {baseline_data['public_burden_per_damage'].std():.2f}%")
        
        print("\nDEFAULT RATE SENSITIVITY:")
        print(f"  Defaults per $B of damage:")
        print(f"    Min: {baseline_data['defaults_per_billion_damage'].min():.3f}")
        print(f"    Mean: {baseline_data['defaults_per_billion_damage'].mean():.3f}")
        print(f"    Max: {baseline_data['defaults_per_billion_damage'].max():.3f}")
        
        # Check for threshold effects
        print("\nTHRESHOLD EFFECTS:")
        high_damage = baseline_data[baseline_data['total_damage_mean'] > baseline_data['total_damage_mean'].quantile(0.75)]
        low_damage = baseline_data[baseline_data['total_damage_mean'] < baseline_data['total_damage_mean'].quantile(0.25)]
        
        if len(high_damage) > 0 and len(low_damage) > 0:
            print(f"  High damage scenarios (top 25%):")
            print(f"    Public burden per damage: {high_damage['public_burden_per_damage'].mean():.2f}%")
            print(f"    Defaults per $B damage: {high_damage['defaults_per_billion_damage'].mean():.3f}")
            
            print(f"  Low damage scenarios (bottom 25%):")
            print(f"    Public burden per damage: {low_damage['public_burden_per_damage'].mean():.2f}%")
            print(f"    Defaults per $B damage: {low_damage['defaults_per_billion_damage'].mean():.3f}")
    
    def check_data_quality(self):
        """Check for potential data issues and odd values."""
        print("\n" + "="*80)
        print("DATA QUALITY CHECKS")
        print("="*80)
        
        issues_found = []
        
        for scenario_name, data in self.scenarios.items():
            df = data['iterations']
            
            # Check 1: Negative values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if (df[col] < 0).any():
                    n_negative = (df[col] < 0).sum()
                    issues_found.append(f"{scenario_name}: {col} has {n_negative} negative values")
            
            # Check 2: NaN values
            if df.isnull().any().any():
                nan_cols = df.columns[df.isnull().any()].tolist()
                issues_found.append(f"{scenario_name}: NaN values in {nan_cols}")
            
            # Check 3: Percentage sums
            pct_cols = [col for col in df.columns if '_pct' in col and 'wind' in col and 'flood' not in col]
            if len(pct_cols) >= 4:
                # Wind percentages should sum to ~100 for non-zero damage years
                damage_years = df[df['wind_total_usd'] > 0]
                if len(damage_years) > 0:
                    pct_sums = damage_years[pct_cols].sum(axis=1)
                    if not np.allclose(pct_sums.mean(), 100, atol=1):
                        issues_found.append(f"{scenario_name}: Wind percentages sum to {pct_sums.mean():.1f}% (should be ~100%)")
            
            # Check 4: Unrealistic ratios
            if df['total_damage_usd'].max() > 0:
                max_damage_year = df.loc[df['total_damage_usd'].idxmax()]
                
                # Check if uninsured is suspiciously high
                if max_damage_year['wind_uninsured_pct'] > 80:
                    issues_found.append(f"{scenario_name}: Very high uninsured rate ({max_damage_year['wind_uninsured_pct']:.1f}%) in max damage year")
                
                # Check if public burden exceeds total damage (shouldn't happen)
                public_burden = (
                    max_damage_year['citizens_residual_deficit_usd'] + 
                    max_damage_year['figa_residual_deficit_usd'] + 
                    max_damage_year['nfip_borrowed_usd']
                )
                if public_burden > max_damage_year['total_damage_usd']:
                    issues_found.append(f"{scenario_name}: Public burden exceeds total damage in max damage year")
        
        if issues_found:
            print(f"\nFound {len(issues_found)} potential issues:\n")
            for issue in issues_found[:20]:  # Show first 20
                print(f"  • {issue}")
            if len(issues_found) > 20:
                print(f"  ... and {len(issues_found) - 20} more issues")
        else:
            print("\n✓ No major data quality issues detected")
    
    def generate_comparison_table(self):
        """Generate a comprehensive comparison table."""
        print("\n" + "="*80)
        print("SCENARIO COMPARISON TABLE")
        print("="*80)
        
        # Select key metrics for comparison
        key_metrics = [
            'model', 'climate', 'policy', 'severity',
            'total_damage_mean', 'defaults_mean', 'prob_any_default',
            'public_burden_mean', 'wind_uninsured_pct', 'prob_nfip_borrowing'
        ]
        
        comparison_df = self.summary_df[key_metrics].copy()
        
        # Format for readability
        comparison_df['total_damage_mean'] = comparison_df['total_damage_mean'] / 1e9
        comparison_df['public_burden_mean'] = comparison_df['public_burden_mean'] / 1e9
        comparison_df['prob_any_default'] = comparison_df['prob_any_default'] * 100
        comparison_df['prob_nfip_borrowing'] = comparison_df['prob_nfip_borrowing'] * 100
        
        comparison_df.columns = [
            'Model', 'Climate', 'Policy', 'Severity',
            'Damage ($B)', 'Defaults (#)', 'Default Prob (%)',
            'Public Burden ($B)', 'Uninsured (%)', 'NFIP Borrow (%)'
        ]
        
        # Sort by damage
        comparison_df = comparison_df.sort_values('Damage ($B)', ascending=False)
        
        print("\nTop 20 scenarios by damage:")
        print(comparison_df.head(20).to_string(index=False))
        
        return comparison_df
    
    def run_full_analysis(self):
        """Run all analyses."""
        self.load_scenarios()
        self.compute_summary_statistics()
        self.analyze_loss_composition()
        self.analyze_institutional_stress()
        self.analyze_policy_effects()
        self.check_nonlinearities()
        self.check_data_quality()
        comparison_table = self.generate_comparison_table()
        
        # Save results
        output_dir = self.results_dir / "analysis"
        output_dir.mkdir(exist_ok=True)
        
        self.summary_df.to_csv(output_dir / "emanuel_comprehensive_summary.csv", index=False)
        comparison_table.to_csv(output_dir / "emanuel_comparison_table.csv", index=False)
        
        print(f"\n{'='*80}")
        print(f"Analysis complete. Results saved to {output_dir}")
        print(f"{'='*80}")
        
        return self


if __name__ == "__main__":
    results_dir = Path("/Users/simonameiler/Documents/work/03_code/repos/systemic_insurance_risk_fl/results/mc_runs")
    
    analyzer = EmanuelAnalyzer(results_dir)
    analyzer.run_full_analysis()
