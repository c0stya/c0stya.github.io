cp -rf img docs/

for f in ./*.typ; do
    pandoc --css "css/github-pandoc.css" --highlight-style tango  --embed-resources  -f typst -t html -s --mathml $f > docs/"${f%.*}".html
done
