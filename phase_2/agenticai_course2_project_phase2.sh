log_file=agenticai_course2_project_phase2.log
> $log_file

echo "Starting script": $1 `date` >> $log_file
python $1 >> $log_file
echo "=====================================" >> $log_file
echo "Completed:" `date` >> $log_file
