import os
import argparse
from boxsdk import JWTAuth, Client

from puzle.utils import return_data_dir


def _return_fname(i):
    return 'download_PS1_PSC.%02d.sh' % i


def init_download_scripts(n_scripts):
    for i in range(n_scripts):
        fname = _return_fname(i)
        if os.path.exists(fname):
            os.remove(fname)
        with open(fname, 'w') as f:
            f.write('#!/bin/bash\n')


def output_cmd(cmd, file_name, i, n_scripts):
    output_idx = i % n_scripts
    fname = _return_fname(output_idx)

    with open(fname, 'a') as f:
        f.write('echo "Downloading %s (%i / 366)"\n' % (file_name, i))
        f.write('%s\n' % cmd)


def open_script_permissions(n_scripts):
    for i in range(n_scripts):
        fname = _return_fname(i)
        os.chmod(fname, 0o775)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-scripts', type=int,
                          help='Number of scripts to split download into.',
                          default=1)
    parser.add_argument('--overwrite', dest='overwrite', action='store_true')
    parser.set_defaults(overwrite=False)

    args = parser.parse_args()
    n_scripts = args.n_scripts
    overwrite = args.overwrite

    data_dir = return_data_dir()
    config_fname = '%s/800207941_6guv3vhr_config.json' % data_dir
    config = JWTAuth.from_settings_file(config_fname)
    client = Client(config)

    _ = client.create_user('User', login=None)
    access_token = config.access_token

    url = "https://northwestern.app.box.com/s/94tv3ry0tlz6tbfmv7aznj5wzzneoqpn"
    shared_folder = client.get_shared_item(url)

    template = 'curl https://api.box.com/2.0/files/{file_id}/content ' \
               '-H "Authorization: Bearer {access_token}" ' \
               '-H "BoxApi: shared_link={url}" -L --output {file_name}'

    init_download_scripts(n_scripts)

    for i, file in enumerate(shared_folder.get_items()):
        if file.type == 'file':
            if not overwrite and os.path.exists(file.name):
                continue
            dict_out = {'access_token': access_token,
                        'url': url,
                        'file_id': file.id,
                        'file_name': file.name}
            cmd = template.format(**dict_out)
            output_cmd(cmd, file.name, i, n_scripts)

    open_script_permissions(n_scripts)


if __name__ == '__main__':
    main()
