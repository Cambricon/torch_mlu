/*---------------------------------------------------------------------------------------------
 * Copyright (c) Microsoft Corporation. All rights reserved.
 *--------------------------------------------------------------------------------------------*/

import { makeStyles } from '@material-ui/core/styles'
import * as React from 'react'

export interface IProps {
  mluInfo: any
}

const useStyles = makeStyles((theme) => ({
  root: {
    border: '1px solid #E0E0E0',
    borderCollapse: 'collapse',
    width: '100%'
  },
  td: {
    borderTop: '1px solid #E0E0E0',
    borderBottom: '1px solid #E0E0E0',
    borderCollapse: 'collapse',
    paddingLeft: 10,
    paddingRight: 10
  },
  nodeTd: {
    fontWeight: 'bold'
  },
  pidTd: {
    fontWeight: 'normal'
  },
  mluTd: {
    fontWeight: 'normal'
  },
  keyTd: {
    fontWeight: 'normal',
    textAlign: 'right'
  },
  valueTd: {
    fontWeight: 'bold'
  }
}))

interface TableCellInfo {
  content: string
  rowspan: number
  cellType: 'node' | 'pid' | 'mlu' | 'key' | 'value'
  last?: boolean
}

function makeTableCellInfo(mluInfo: any): TableCellInfo[][] {
  const rows: TableCellInfo[][] = []
  let curr_row: TableCellInfo[] = []
  rows.push(curr_row)
  Object.keys(mluInfo.data).forEach(function (node_name) {
    const node_cell = {
      content: node_name,
      rowspan: 0,
      cellType: 'node' as const
    }
    const i = rows.length
    curr_row.push(node_cell)
    Object.keys(mluInfo.data[node_name]).forEach(function (pid) {
      const pid_cell = { content: pid, rowspan: 0, cellType: 'pid' as const }
      const i = rows.length
      curr_row.push(pid_cell)
      Object.keys(mluInfo.data[node_name][pid]).forEach(function (mlu) {
        const mlu_cell = { content: mlu, rowspan: 0, cellType: 'mlu' as const }
        const i = rows.length
        curr_row.push(mlu_cell)
        Object.keys(mluInfo.data[node_name][pid][mlu]).forEach(function (
          key_name
        ) {
          curr_row.push({
            content: key_name,
            rowspan: 1,
            cellType: 'key' as const
          })
          const value: string = mluInfo.data[node_name][pid][mlu][key_name]
          curr_row.push({
            content: value,
            rowspan: 1,
            cellType: 'value' as const
          })
          curr_row = []
          rows.push(curr_row)
        })
        mlu_cell.rowspan = rows.length - i
      })
      pid_cell.rowspan = rows.length - i
    })
    node_cell.rowspan = rows.length - i
  })
  rows.pop()
  return rows
}

export const MluInfoTable: React.FC<IProps> = (props) => {
  const classes = useStyles()
  interface TableCellInfo {
    content: string
    rowspan: number
    cellType: 'node' | 'pid' | 'mlu' | 'key' | 'value'
  }

  const rows = React.useMemo(() => makeTableCellInfo(props.mluInfo), [
    props.mluInfo
  ])

  const cellToClass = {
    node: classes.nodeTd,
    pid: classes.pidTd,
    mlu: classes.mluTd,
    key: classes.keyTd,
    value: classes.valueTd
  }

  const renderCell = function (info: TableCellInfo) {
    let cellClass = cellToClass[info.cellType]
    let content = info.cellType == 'key' ? info.content + ':' : info.content
    return (
      <td className={classes.td + ' ' + cellClass} rowSpan={info.rowspan}>
        {content}
      </td>
    )
  }

  return (
    <table className={classes.root}>
      {rows.map((row) => (
        <tr>{row.map(renderCell)}</tr>
      ))}
    </table>
  )
}
