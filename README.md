# Landing Page A/B Test: Does the Redesign Increase Conversion?

**Recommendation: keep the original page.** The redesigned page converted at 11.88% vs. 12.04% for the original (−0.16 percentage points, p = 0.19). The 95% confidence interval for the difference, −0.39 to +0.08 pp, rules out any lift larger than 0.08 pp.

![Conversion by group and by day](output/conversion_results.png)

## Data

Two public CSVs from the Udacity Data Analyst Nanodegree "Analyze A/B Test Results" project:

- `ab_data.csv`: 294,478 events from a 22-day experiment (Jan 2–24, 2017), with user ID, timestamp, assigned group, page shown, and whether the user converted
- `countries.csv`: country (US, UK, CA) for each user

## Pipeline

1. **Load** both CSVs into SQLite.
2. **Check data integrity in SQL** (`sql/queries.sql`). The data had no nulls. However, 3,893 rows (1.3%) showed a user a page that contradicted their assignment, such as a control user seeing the new page.
3. **Build a clean analytical table in SQL.** The table drops misassigned rows, excludes users exposed to both arms, and keeps each user's first exposure. It joins country, leaving **290,584 users**, one row each.
4. **Compute metrics in SQL** by group, country, and day.
5. **Run the statistics in Python** (`analysis.py`), as summarized below.

## Results

| Check | Method | Result |
| --- | --- | --- |
| Traffic split | Sample-ratio-mismatch chi-square test vs. 50/50 | p = 0.95, so the split was balanced and the randomization looks sound |
| Effect on conversion | Two-proportion z-test | z = −1.31, p = 0.19 (two-sided); no significant difference |
| Size of effect | 95% CI on the difference | −0.39 to +0.08 pp (relative lift −1.3%) |
| Was the test big enough? | Minimum detectable effect, 80% power, α = 0.05 | 0.34 pp (2.8% relative); a meaningful lift would likely have been detected |
| Different effect by country? | Logistic regression, treatment × country, likelihood-ratio test | p = 0.29; no evidence the new page works better in any one country |
| Novelty or learning effects | Daily conversion by arm | No sustained gap between arms over the 22 days |

## Why the recommendation holds

The test was well powered to detect a 2.8% relative lift, and the confidence interval excludes any meaningful gain. Shipping the redesign would add engineering and design cost with no measurable benefit to conversion. A smaller effect could only be detected with a longer, larger test, and that test is only worth running if a sub-0.1 pp lift would matter to the business.

## Run it

```bash
pip install -r requirements.txt
python analysis.py
```

The script prints each SQL check and writes `output/results.json` and `output/conversion_results.png`.

## Files

```
sql/queries.sql      integrity checks, clean-table build, metrics (named queries)
analysis.py          loads data, runs the SQL, statistics, chart
data/                raw CSVs
output/              results.json, conversion_results.png, run_log.txt
```
