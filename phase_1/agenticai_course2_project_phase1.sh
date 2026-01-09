log_file=agenticai_course2_project_phase1.log
> $log_file
for script in `ls -tr1 *.py`
do 
echo "Starting script": $script `date` >> $log_file
python $script >> $log_file
echo "=====================================" >> $log_file
done
rm chunk*.csv embed*.csv
echo "Completed:" `date` >> $log_file
