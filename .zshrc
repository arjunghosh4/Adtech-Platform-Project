export SPARK_HOME=$HOME/spark-3.3.2
export PATH=$SPARK_HOME/bin:$PATH
export PATH=${PATH}:/Users/arjunghosh/mongodb-macos-aarch64-7.0.14/bin
export PATH=${PATH}:/Users/arjunghosh/mongosh-2.3.1-darwin-arm64/bin 
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.zsh' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

