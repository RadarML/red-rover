PROJECT=arrow PROJECT_PATH=processing/arrow make html
PROJECT=processing make html
PROJECT=collect make html
PROJECT=deepradar PROJECT_PATH=../deep-radar make html
 for i in `ls _build`; do cp -r _build/$i/html _export/$i; done
 