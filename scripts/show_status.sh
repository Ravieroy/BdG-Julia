#! /bin/bash

#----For color-----

txtred=$(tput setaf 1)
txtylw=$(tput setaf 3)
txtblu=$(tput setaf 4)
txtrst=$(tput sgr0)

cd "../logs" || exit

check_status(){
    val=$(grep -c "Completed" "$f")    
    if [ "$val" -eq 1 ]; then
        is_completed=true
        is_running=false
    else
        is_running=true
        is_completed=false
    fi
}

print_completed_status(){
    if [[ "$is_completed" == true ]]; then
        seed=$(grep "seed" "$f")
        message=$(grep Completed "$f" | tail -1) 
        echo "${txtblu} $seed : $message ${txtrst}"
    fi
}

print_running_status(){
    if [[ "$is_running" == true ]]; then
        seed=$(grep "seed" "$f")
        message=$(grep Running "$f" | tail -1) 
        echo "${txtylw} $seed : $message ${txtrst}"
    fi
}

is_any_job_running(){
    n_total_files=$(find . -name "*.log" -type f | wc -l) 
    n_completed=$(grep -R "Completed" . | wc -l)

    if [[ "$n_total_files" == "$n_completed" ]]; then
        echo "${txtylw}No running jobs${txtrst}"
        exit
    fi
}

# Function to display script usage
usage() {
     echo "${txtylw}Description${txtrst}: Show status of completed and running jobs"
     echo "${txtylw}Usage${txtrst}: $0 [OPTIONAL FLAG]"
     echo "${txtylw}Flags${txtrst}:"
     echo " -h,     Display this help message"
     echo " -r,     Show status of only the running jobs" 
}

# main driver script
if [ $# -eq 0 ]; then
    for f in *.log; do
        check_status f
        print_running_status
        print_completed_status
    done
else
    while getopts "hr" flag; do
        case $flag in
            h)
                usage
                ;;
                
            r) #show only running jobs
                is_any_job_running
                for f in *.log; do
                    check_status f
                    print_running_status
                done
 
                ;;
            \?)
                echo "${txtred}ERROR: Wrong usage${txtrst}"
                usage 
                ;;
        esac
    done
fi


