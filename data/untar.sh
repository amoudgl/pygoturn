while read -r line
do
    tar -xvf $line
done < <(find . -name "*.tar")