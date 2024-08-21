import boto3
import os
import logging
import subprocess
import math
import json

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    #AWS interaction
    s3_client = boto3.client('s3')
    download_path = os.path.join("/tmp", os.path.basename(key))
    s3_client.download_file(bucket, key, download_path)
    logger.info("Download complete")

    output_dir = video_splitting_cmdline(download_path)
    logger.info("Split complete")
    #Uploading images
    first_image = next((f for f in os.listdir(output_dir) if f.endswith(".jpg")), None)
    if first_image:
        s3_client.upload_file(os.path.join(output_dir, first_image), '1216967843-stage-1', first_image)
        logger.info(f"Uploaded {first_image} to '1216967843-stage-1' bucket")

    logger.info("Upload complete")
    invoke_face_recognition('1216967843-stage-1', first_image)
    logger.info("Face recognition invoked")
    os.remove(download_path)
    
    return {
        'statusCode': 200,
        'body': 'Video processed and first image uploaded successfully'
    }
#upon response invoke face recognition program
def invoke_face_recognition(output_bucket, file_name):
    lambda_client = boto3.client('lambda')
    payload = {
        'output_bucket': output_bucket,
        'file_name': file_name
    }
    lambda_client.invoke(FunctionName='face-recognition', InvocationType='Event', Payload=json.dumps(payload))
#split video into images
def video_splitting_cmdline(video_filename):
    filename = os.path.basename(video_filename)
    outdir = os.path.splitext(filename)[0]
    outdir = os.path.join("/tmp", outdir)
    output_dir = outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    base_filename = os.path.splitext(filename)[0]
    split_cmd = '/usr/bin/ffmpeg -ss 0 -r 1 -i ' + video_filename + ' -vf fps=1/10 -start_number 0 -vframes 10 ' + outdir + "/" + base_filename + '.jpg -y'
    try:
        subprocess.check_call(split_cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.output)

    fps_cmd = 'ffmpeg -i ' + video_filename + ' 2>&1 | sed -n "s/.*, \\(.*\\) fp.*/\\1/p"'
    fps = subprocess.check_output(fps_cmd, shell=True).decode("utf-8").rstrip("\n")
    fps = math.ceil(float(fps))
    return outdir
