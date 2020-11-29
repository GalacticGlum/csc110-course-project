#!/bin/bash
#
# Downloads the ArchiveTeam TwitterStream data torrent files from The Internet Archive.
# Requires:
#   - internetarchive CLI (install with pip)
#   - GNU parallel
#
# Run with: bash download_twitter_data_torrent.sh

ia search "collection:twitterstream" --itemlist |
    sort -r |
    parallel "ia download {}" --glob="*.torrent" --no-directories