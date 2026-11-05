# Deploying to Google Cloud Run

These commands deploy the API as a Cloud Run service named `dog-breed-api` in `us-west1`.
Cloud Build builds the image from the `Dockerfile`, which downloads the model from
[jakobmp/dog-breed-densenet121](https://huggingface.co/jakobmp/dog-breed-densenet121) and
checks its SHA-256. The service scales to zero when idle, so it costs nothing between requests.

The commands are for PowerShell on Windows. Run them from the repo root.

## 1. Install and log in to the gcloud CLI

Install the Google Cloud CLI from https://cloud.google.com/sdk/docs/install, then open a new
terminal and log in:

```powershell
gcloud auth login
```

## 2. Create a project and link billing

Project IDs are global, so pick a unique one (lowercase letters, digits and hyphens):

```powershell
$PROJECT = "dog-breed-classifier-api"
gcloud projects create $PROJECT
gcloud config set project $PROJECT
gcloud config set run/region us-west1
```

Cloud Run needs an open billing account, even within the free tier. Find yours and link it
(if its `OPEN` column says `False`, reopen it first in the console under Billing):

```powershell
gcloud billing accounts list
$BILLING = "XXXXXX-XXXXXX-XXXXXX"   # the ACCOUNT_ID from the list
gcloud billing projects link $PROJECT --billing-account $BILLING
```

To use an existing project instead, skip `projects create` and just run `gcloud config set project`.

## 3. Enable the APIs

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com billingbudgets.googleapis.com
```

## 4. Set a budget alert

```powershell
gcloud billing budgets create --billing-account $BILLING --display-name "dog-breed-api" `
  --budget-amount 5USD `
  --threshold-rule percent=0.5 --threshold-rule percent=0.9 --threshold-rule percent=1.0
```

Google emails the billing account's admins at 50%, 90% and 100% of $5 in a month.
**Budget alerts only notify; they never stop spending.** The real cost limit is the deploy
settings below: `--max-instances 1` means at most one container ever runs, and
`--min-instances 0` means none runs while the service is idle. You can also manage budgets
in the console under Billing > Budgets & alerts.

## 5. Deploy

```powershell
gcloud run deploy dog-breed-api --source . --region us-west1 `
  --allow-unauthenticated `
  --min-instances 0 --max-instances 1 `
  --memory 1Gi --cpu 1 --concurrency 4 `
  --set-env-vars "^@^ALLOWED_ORIGINS=https://dog-breed-classifier-frontend.vercel.app,http://localhost:3000"
```

- `--source .` uploads the repo (minus the files in `.gcloudignore`) and builds it with Cloud Build.
  The first time, gcloud asks to create an Artifact Registry repository named
  `cloud-run-source-deploy`; answer `Y`.
- The model URL is the default `MODEL_URL` in the `Dockerfile`. `gcloud run deploy --source`
  can't pass build args, so to serve a different model file, change that line (and
  `MODEL_SHA256`) and redeploy.
- `^@^` tells gcloud to split variables on `@` instead of commas, so the comma-separated
  origin list stays one value.
- `--allow-unauthenticated` makes the API public, which the frontend needs.
- `--concurrency 4` matches gunicorn's 4 threads.

The build takes a few minutes. When it finishes, gcloud prints the service URL, like
`https://dog-breed-api-xxxxxxxxxx-uw.a.run.app`. To redeploy after a code change, run the
same command again.

## 6. Test the deployed service

```powershell
$URL = gcloud run services describe dog-breed-api --region us-west1 --format "value(status.url)"
$IMG = "path/to/a/dog-photo.jpg"   # any photo of a dog

curl.exe -s "$URL/"                                   # Dog Breed Classifier API 1.0.0
curl.exe -s "$URL/health"                             # {"status":"ok"}
curl.exe -s -F "image=@$IMG" "$URL/predict"           # {"accuracy":...,"breed":"Beagle","top_5":[...]}
curl.exe -s -F "other=x" "$URL/predict"               # 400: no image field
curl.exe -s -F "image=@DEPLOY.md" "$URL/predict"      # 400: not an image
```

Check CORS: the first should print an `access-control-allow-origin` header, the second nothing.

```powershell
curl.exe -s -D - -o NUL -H "Origin: https://dog-breed-classifier-frontend.vercel.app" "$URL/health" | Select-String access-control-allow-origin
curl.exe -s -D - -o NUL -H "Origin: https://example.com" "$URL/health" | Select-String access-control-allow-origin
```

### Cold start

After about 15 minutes with no requests, Cloud Run shuts the container down. Time the next
request, which has to start a container and load the model, then a warm one:

```powershell
curl.exe -s -o NUL -w "cold: %{time_total}s`n" "$URL/health"
curl.exe -s -o NUL -w "warm: %{time_total}s`n" "$URL/health"
```

The frontend calls `/health` when the page loads, so the container is usually warm by the
time someone uploads a photo.

### Logs

```powershell
gcloud run services logs read dog-breed-api --region us-west1 --limit 30
```

Each prediction logs one line, such as `Predicted Beagle (0.981) in 40 ms`. Client IP
addresses are not logged by the app. Cloud Run's own request logs do record them.

## Costs

- **Cloud Run:** billed only while handling requests (the default "request-based" billing),
  and light use stays within the monthly free tier. With `--max-instances 1`, the worst case
  is one container busy all month.
- **Cloud Build:** each deploy runs one build, well within the free build minutes for
  occasional deploys.
- **Artifact Registry:** each deploy stores a new image of a few hundred MB. Storage beyond
  the free amount costs a few cents per GB per month. To keep only recent images, delete old
  ones in the console under Artifact Registry > `cloud-run-source-deploy`, or set a cleanup
  policy there.

See https://cloud.google.com/run/pricing for current prices.

## Troubleshooting

- **Build fails with a permission error** (for example on `storage.objects.get` or
  "does not have permission to write logs"): newer projects build with the Compute Engine
  default service account, which may need the Cloud Run builder role:

  ```powershell
  $NUMBER = gcloud projects describe $PROJECT --format "value(projectNumber)"
  gcloud projects add-iam-policy-binding $PROJECT `
    --member "serviceAccount:$NUMBER-compute@developer.gserviceaccount.com" --role roles/run.builder
  ```

- **Build fails with `SHA-256 mismatch`:** the file at `MODEL_URL` isn't the expected model.
  Check the Hugging Face repo and the `Dockerfile`'s `MODEL_URL` and `MODEL_SHA256`.
- **`/health` returns 503:** the model didn't load. Check the logs for `Could not load model`.
- **The browser shows a CORS error:** the frontend's origin isn't in `ALLOWED_ORIGINS`. Redeploy
  with the right list, or update it without rebuilding:

  ```powershell
  gcloud run services update dog-breed-api --region us-west1 --update-env-vars "^@^ALLOWED_ORIGINS=https://...,http://localhost:3000"
  ```

## Removing everything

```powershell
gcloud run services delete dog-breed-api --region us-west1
gcloud artifacts repositories delete cloud-run-source-deploy --location us-west1
```

Or delete the whole project with `gcloud projects delete $PROJECT`.
