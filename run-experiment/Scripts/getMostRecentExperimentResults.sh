#!/bin/bash
dir = $(ls -td Logs/* | head -1)
echo ${dir}/training_session.log
