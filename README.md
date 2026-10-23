# Vendor Invoice Intelligence

A complete local Python project for estimating freight costs and prioritizing vendor invoices for review. It includes a Streamlit dashboard, two trained machine learning models, normalized SQLite data, CSV batch analysis, and reproducible setup and training commands.

This is an independent implementation of the business problem demonstrated in [Tech Classes' Invoice Intelligence Machine Learning Project](https://www.youtube.com/watch?v=0DIuUEZPIj4). The creator's source code has not been copied or purchased.

## Start on Windows

Project location: `D:\Python\vendor_invoice_intelligence`

1. Install Python **3.10 or newer** if it is not already available. Make sure Python is added to PATH.
2. Open the project folder and double-click **Start.bat**.
3. On the first run, setup creates an isolated `.venv`, installs dependencies, generates demonstration data, and trains both models. This step needs an internet connection to download Python packages.
4. Open **http://localhost:8501** in your browser.

You can also launch from PowerShell:

```powershell
Set-Location D:\Python\vendor_invoice_intelligence
python bootstrap.py
```

The application runs locally and can work offline after setup. Keep the terminal open while using the dashboard; press `Ctrl+C` to stop the server.

**Setup.bat** prepares the environment, data, and models. **Verify.bat** runs the project's verification checks. **Start.bat** starts the application and performs any missing initial setup.

## What the application does

- **Freight prediction:** estimate expected freight from invoice and shipment details using a Random Forest regressor.
- **Invoice review:** inspect a model risk score together with explicit review policies and the reasons they trigger.
- **Batch analysis:** upload a CSV containing up to **5,000 rows**, analyze valid invoices, and download results. Invalid rows remain in the output with validation errors so they can be corrected.
- **Dashboard:** explore demonstration invoices, costs, model evaluation, and review outcomes.

All monetary amounts use **USD**. An invoice marked for review needs a human assessment; its status does not establish fraud or authorize a payment decision. Model scores are **uncalibrated** and should not be interpreted as verified probabilities of fraud. Explicit review policies are evaluated separately from the classifier, making their reasons visible independently of the model score.

## Demonstration data and model limits

The project starts with **generated demonstration data**, not real vendor transactions or the original video's database. The original [free dataset is offered by the creator on Topmate](https://topmate.io/ayushi_mishra/1981139); its public listing describes a SQLite database, but no direct download was available without going through the creator's acquisition flow. This project does not require that dataset or a paid resource to run.

The generated records support a repeatable end-to-end demonstration: normalized SQLite tables, SQL aggregation, feature preparation, model training, serialization, and inference. Their review labels reflect the demonstration's assumptions. Evaluation uses a **chronological 80% training / 20% holdout split**, and the reported scores measure performance on generated data only. They are not evidence of performance on real invoices. Freight regression and risk classification use separate Random Forest models.

Before real operational use, replace the demonstration data with appropriately prepared historical data, define review labels with domain experts, retrain, and validate on a representative holdout set. The original database may need a mapping to this project's schema; downloading it does not automatically make its tables compatible.

## Batch CSV format

Upload a UTF-8 CSV with a header row. Each row represents one invoice.

| Column | Required | Meaning / default |
| --- | --- | --- |
| `quantity` | Yes | Invoice quantity; greater than zero |
| `invoice_amount` | Yes | Invoice value in USD; greater than zero |
| `weight_kg` | Yes | Shipment weight in kilograms; greater than zero |
| `distance_km` | Yes | Shipment distance in kilometres; greater than zero |
| `freight_cost` | Yes | Actual freight charged in USD; zero or greater |
| `line_total` | Yes | Sum of item amounts in USD; greater than zero |
| `line_count` | No | Number of invoice lines; defaults to `3` |
| `shipping_mode` | No | Shipping mode; defaults to `Standard` |
| `region` | No | Shipment region; defaults to `North` |
| `payment_terms_days` | No | Payment terms in days; defaults to `30` |

Use the sample CSV supplied by the application as the authoritative template, including accepted categorical values. Do not include currency symbols or thousands separators in numeric fields.

```csv
quantity,invoice_amount,weight_kg,distance_km,freight_cost,line_total
100,2500,80,450,120,2500
25,900,20,125,35,900
```

Batch results preserve invalid rows and identify their validation errors. Correct those rows and upload them again to obtain predictions.

## Commands

Run these commands from the project folder after the initial setup:

```powershell
# Create any missing demonstration database, models, and sample files.
.\.venv\Scripts\python.exe engine.py --setup

# Retrain the models and refresh their evaluation results.
.\.venv\Scripts\python.exe engine.py --train

# Launch the dashboard using the isolated Python environment.
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The launcher handles setup automatically. When running the app directly with Streamlit, use the project folder as the current directory so `.streamlit/config.toml` is loaded.

## Project structure

```text
vendor_invoice_intelligence/
  app.py                 Streamlit dashboard
  engine.py              SQLite, validation, features, training, and inference
  bootstrap.py           Isolated environment setup and application launcher
  Start.bat              Windows start shortcut
  Setup.bat              Windows setup shortcut
  Verify.bat             Windows verification shortcut
  requirements.txt       Python dependencies
  .streamlit/config.toml Local server and dark theme configuration
```

Setup also creates local data, model artifacts, and the `.venv` environment. Model artifacts are generated locally; only load serialized model files from sources you trust.

## Troubleshooting

- **Python is not recognized:** install Python 3.10+ and enable its PATH option, then reopen the terminal.
- **Dependency installation fails:** check internet access and rerun `Setup.bat`. Do not close the setup terminal while packages are being installed.
- **The page does not open:** keep the launcher terminal running and visit http://localhost:8501 manually.
- **Port 8501 is in use:** stop the other local instance, or use `.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502` and open http://localhost:8502.
- **A CSV row is rejected:** inspect its validation message and compare the input to the downloadable template. A rejected row has no prediction.

The server binds to `127.0.0.1` by default. Streamlit usage telemetry is disabled in the included configuration.

## Included verification

Ten automated checks cover SQL aggregation, repeatable initialization, model evaluation and reload,
input validation, CSV errors, row preservation, independent policy flags, persistent review history,
CSV export, and a Streamlit AppTest walkthrough of all six pages. Run `Verify.bat` to repeat them.

The project was verified with Python 3.12. A snapshot of the installed package versions is saved
in `requirements-lock.txt` for reproducing that environment. Regular setup uses `requirements.txt`.
The dashboard uses local fallback fonts and needs no internet connection after package installation.
