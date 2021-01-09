import os

# train_dir = './res/'

def progress(percent, width=50):
    '''Progress printing function'''
    if percent >= 100:
        percent = 100

    show_str = ('[%%-%ds]' % width) % (int(width * percent / 100) * "#")  # Netsed use of string splicing
    print('\r%s %d%% ' % (show_str, percent), end='')

def is_valid_jpg(jpg_file):
    with open(jpg_file, 'rb') as f:
        f.seek(-2, 2)
        buf = f.read()
        f.close()
        return buf ==  b'\xff\xd9'  # Determine whether the .jpg contains the end field

myFile = open("corrupted_images.txt", 'a')
fDir = '/home/ubuntu/.jupyter/MyNotebooks/imagenet-a/imagenet-a/'
# fSubDir = ["n01498041/", "n01531178/", "n01534433/"]
fSubDir = os.listdir(fDir)

for i in range(0, len(fSubDir)):
  train_dir = fDir + fSubDir[i] + '/'

  data_size = len([lists for lists in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, lists))])
  recv_size = 0
  incompleteFile = 0
  print('file tall : %d' % data_size)

  for file in os.listdir(train_dir):
      if os.path.splitext(file)[1].lower() == '.jpg':
          ret = is_valid_jpg(train_dir + file)
          if ret == False:
              incompleteFile = incompleteFile + 1
              print(train_dir + file + '\n')
              myFile.write(train_dir + file + '\n')
              os.remove(train_dir + file)

      recv_per = int(100 * recv_size / data_size)
      progress(recv_per, width=30)
      recv_size = recv_size + 1

  progress(100, width=30)
  print('\nincomplete file : %d' % incompleteFile)
myFile.close()
