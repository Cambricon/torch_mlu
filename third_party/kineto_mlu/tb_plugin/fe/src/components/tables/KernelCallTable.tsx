import * as React from 'react'
import { Table, TableColumnsType } from 'antd'

import { TableRow } from '../DiffOverview'

export interface IProps {
  data: TableRow
}

export interface KernelCallRow {
  key: number
  
  baselineKernelName: string
  baselineKernelCalls: number
  expKernelName: string
  expKernelCalls: number
}

export const KernelCallTable = (props: IProps) => {
  const {data} = props
  const {baselineKernelNameListStr, baselineKernelCallsListStr, expKernelNameListStr, expKernelCallsListStr} = data

  // liststr to list
  const baselineKernelNameList = (baselineKernelNameListStr ?? '').trim().split(';').filter(item => item != '').map((x) => x.trim())
  const baselineKernelCallsList = (baselineKernelCallsListStr ?? '').trim().split(';').filter(item => item != '').map((x) => Number(x.trim()))
  const expKernelNameList = (expKernelNameListStr ?? '').trim().split(';').filter(item => item != '').map((x) => x.trim())
  const expKernelCallsList = (expKernelCallsListStr ?? '').trim().split(';').filter(item => item != '').map((x) => Number(x.trim()))

  const baseKernelCallColumns: TableColumnsType<KernelCallRow> = [
    {
      title: 'Baseline Kernel Name',
      dataIndex: 'baselineKernelName',
      key: 'baselineKernelName',
      sorter: (a: KernelCallRow, b: KernelCallRow) => a.baselineKernelName?.localeCompare(b.baselineKernelName || '')
    },
    {
      title: 'Baseline Kernel Calls',
      dataIndex: 'baselineKernelCalls',
      key: 'baselineKernelCalls',
      sorter: (a: KernelCallRow, b: KernelCallRow) => a.baselineKernelCalls! - b.baselineKernelCalls!
    },
    {
      title: 'Exp Kernel Name',
      dataIndex: 'expKernelName',
      key: 'expKernelName',
      sorter: (a: KernelCallRow, b: KernelCallRow) => a.expKernelName?.localeCompare(b.expKernelName || '')
    },
    {
      title: 'Exp Kernel Calls',
      dataIndex: 'expKernelCalls',
      key: 'expKernelCalls',
      sorter: (a: KernelCallRow, b: KernelCallRow) => a.expKernelCalls! - b.expKernelCalls!
    }
  ]

  let tableDataSource: KernelCallRow[] = []
  let baselineLength = (baselineKernelNameList?.length || 0)
  let expLength = (expKernelNameList?.length || 0)
  let maxLength: number = baselineLength > expLength ? baselineLength : expLength
  for (let i = 0; i < maxLength; i++) {
    tableDataSource.push({
      key: i,
      baselineKernelName: baselineKernelNameList? baselineKernelNameList[i] : '',
      baselineKernelCalls: baselineKernelCallsList? baselineKernelCallsList[i] : 0,
      expKernelName: expKernelNameList? expKernelNameList[i] : '',
      expKernelCalls: expKernelCallsList? expKernelCallsList[i] : 0
    })
  }

  return (
    <Table
      dataSource={tableDataSource}
      columns={baseKernelCallColumns}
    />
  )
}