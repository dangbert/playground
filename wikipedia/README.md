
## Converting wikipedia download to plain text:
[article](https://blog.afterthedeadline.com/2009/12/04/generating-a-plain-text-corpus-from-wikipedia/)

[Full list of wikipedia downloads / languages](https://dumps.wikimedia.org/backup-index.html)

````bash
# where pt-data is the extracted version of the Portugues wikipedia download
mkdir pt-out
python2 wikipedia2text/xmldump2files.py pt-data.xml pt-out/

# step 4 (better than the into8.sl command used in the article)
cd wikipedia2text
find ../pt-out/ -type f -iname '*.txt' | xargs -L 1 realpath > ../pt-out/pt.files
# this one could take 12+ hours)
cat ../pt-out/pt.files | xargs -L 1 -P 8 -I@ bash -c  'php ./wiki2xml/php/wiki2xml_command.php "@" "@.xml"'
# run this in another terminal at the same time (kills php processes that hang)
java -jar sleep.jar watchthem.sl

# last step (after above completes):
python2 wikiextract.py ../pt-out/ ../pt_plaintext.txt
````

### note:
This process doesn't maintain accents :( (those characters disappear)
**Maybe better to download a set of ebooks as a corpus** (or use a different script for processing wikipedia)